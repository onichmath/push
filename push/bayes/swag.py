import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from typing import *
from collections import defaultdict
from push.bayes.infer import Infer
from push.particle import Particle, ParticleView
from push.bayes.utils import flatten, unflatten_like
from push.lib.utils import detach_to_cpu
import torch.optim.lr_scheduler as lr_scheduler
import os


# =============================================================================
# Helper functions
# =============================================================================
# TODO: pass this in or use .gin configs instead of global var for saving
SAVE_WEIGHTS = True


def _set_swag_state_helper(particle, mom1, mom2, cov_mat_sqrt):
    """Helper function to set SWAG state and model weights to mean."""
    # Set the SWAG moments
    particle.state[particle.pid]["mom1"] = mom1
    particle.state[particle.pid]["mom2"] = mom2
    particle.state[particle.pid]["cov_mat_sqrt"] = cov_mat_sqrt
    
    # Set model parameters to the mean for deterministic evaluation
    for param, mean_param in zip(particle.module.parameters(), mom1):
        param.data.copy_(mean_param.data)


def mk_scheduler(optim):
    """
    Returns Adam optimizer.

    Args:
        params: Model parameters.

    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    # return lr_scheduler.StepLR(optim, step_size=200, gamma=0.1)
    # return lr_scheduler.ExponentialLR(optim, gamma=0.1, last_epoch=-1, verbose='deprecated')
    return lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=1.0, total_iters=1)


def save_swag_moments(particle: Particle, save_dir: str, epoch: int) -> None:
    """Save SWAG moments (mom1, mom2, cov_mat_sqrt) to disk.

    Args:
        particle (Particle): The particle whose moments to save
        save_dir (str): Directory to save the moments
        epoch (int): Current epoch number
    """
    if not SAVE_WEIGHTS:
        return

    pid = particle.pid
    state = particle.state[pid]

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        state["mom1"], os.path.join(save_dir, f"particle{pid}_epoch{epoch}_mom1.pt")
    )
    torch.save(
        state["mom2"], os.path.join(save_dir, f"particle{pid}_epoch{epoch}_mom2.pt")
    )
    torch.save(
        state["cov_mat_sqrt"],
        os.path.join(save_dir, f"particle{pid}_epoch{epoch}_cov_mat_sqrt.pt"),
    )


def save_particle_weights(particle: Particle, save_dir: str, epoch: int) -> None:
    """Save particle weights to disk.

    Args:
        particle (Particle): The particle whose weights to save
        save_dir (str): Directory to save the weights
        epoch (int): Current epoch number
    """
    if not SAVE_WEIGHTS:
        return

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        particle.module.state_dict(),
        os.path.join(save_dir, f"particle{particle.pid}_epoch{epoch}_weights.pt"),
    )


def save_all_particle_weights(
    particle: Particle, other_pids: List[int], save_dir: str, epoch: int
) -> None:
    """Save weights for all particles in the ensemble.

    Args:
        particle (Particle): The lead particle
        other_pids (List[int]): List of other particle IDs
        save_dir (str): Directory to save the weights
        epoch (int): Current epoch number
    """
    if not SAVE_WEIGHTS:
        return

    save_particle_weights(particle, save_dir, epoch)

    for pid in other_pids:
        fut = particle.send(pid, "GET_MODULE_STATE")
        state_dict = fut.wait()

        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            state_dict,
            os.path.join(save_dir, f"particle{pid}_epoch{epoch}_weights.pt"),
        )


# =============================================================================
# Swag Training
# =============================================================================


def _swag_step(
    particle: Particle,
    loss_fn: Callable,
    data: torch.Tensor,
    label: torch.Tensor,
    *args: Any,
) -> None:
    """Calls one SWAG particle's step function.

    Args:
        particle (Particle): SWAG particle.
        loss_fn (Callable): Loss function.
        data (torch.Tensor): Input data.
        label (torch.Tensor): Ground truth labels.
        *args: Additional arguments.
    """
    particle.step(loss_fn, data, label, *args)


def update_theta(state, state_sq, state_cov_mat_sqrt, param, param_sq, n, cov_mat_rank):
    """Updates the first and second moments and iterates the number of parameter settings averaged.

    Args:
        state: First moment.
        state_sq: Second moment.
        param: Parameters.
        param_sq: Squared parameters.
        n (int): Number of iterations.
    """
    for st, st_sq, st_cov_mat_sqrt, p, p_sq in zip(
        state, state_sq, state_cov_mat_sqrt, param, param_sq
    ):
        st.data = (st.data * n + p.data) / (n + 1)
        st_sq.data = (st_sq.data * n + p_sq.data) / (n + 1)
        dev = (p.data - st.data).view(-1, 1)
        dev = dev.to(st_cov_mat_sqrt.device)
        st_cov_mat_sqrt.data = torch.cat((st_cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)
        # remove first column if we have stored too many models
        if (n + 1) > cov_mat_rank:
            st_cov_mat_sqrt.data = st_cov_mat_sqrt.data[1:, :]


def _swag_swag(particle: Particle, reset: bool, cov_mat_rank: int) -> None:
    """Initializes or updates moments for SWAG.

    Args:
        particle (Particle): SWAG particle.
        reset (bool): Whether to reset or update moments.
        cov_mat_rank (int): Maximum rank of low rank plus diagonal covariance matrix
    """
    state = particle.state
    if reset:
        state[particle.pid] = {
            "mom1": [param.clone() for param in particle.module.parameters()],
            "mom2": [
                param.clone() * param.clone() for param in particle.module.parameters()
            ],
            "cov_mat_sqrt": [
                torch.zeros((0, param.numel())).to(particle.device)
                for param in particle.module.parameters()
            ],
        }
        params = [param for param in particle.module.parameters()]
    else:
        params = [param for param in particle.module.parameters()]
        params_sq = [param * param for param in particle.module.parameters()]
        update_theta(
            state[particle.pid]["mom1"],
            state[particle.pid]["mom2"],
            state[particle.pid]["cov_mat_sqrt"],
            params,
            params_sq,
            state["n"],
            cov_mat_rank,
        )
        state["n"] += 1


def _bootstrap_mswag_particle(
    particle: Particle,
    dataloader: DataLoader,
    loss_fn: Callable,
    cov_mat_rank: int,
    pretrain_epochs: int,
    swag_epochs: int,
    swag_pids: List[int],
    val_dataloader=None,
    val_corrupt_dataloader=None,
    save_metrics: bool = True,
    optimizer_name: str = "unknown",
) -> None:
    """Bootstrap MSWAG particle training with metric saving."""
    
    print(f"🐛 DEBUG: _bootstrap_mswag_particle called with save_metrics={save_metrics}, optimizer_name={optimizer_name}")
    
    def evaluate_model(model, dataloader, device):
        """Evaluate model on a dataloader and return accuracy."""
        if dataloader is None:
            return 0.0
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        model.train()
        return correct / total if total > 0 else 0.0
    
    def save_epoch_metrics(epoch, phase, train_loss, val_acc=None, val_corrupt_acc=None):
        """Save metrics for a given epoch."""
        print(f"🐛 DEBUG: Bootstrap save_epoch_metrics called - epoch={epoch}, phase={phase}, save_metrics={save_metrics}, optimizer_name={optimizer_name}")
        
        if not save_metrics:
            print(f"🐛 DEBUG: save_metrics is False, skipping save")
            return
            
        metrics = {
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_corrupt_acc': val_corrupt_acc
        }
        
        # Create results directory structure
        results_dir = f"results/{optimizer_name}/evaluation_results"
        print(f"🐛 DEBUG: Creating directory: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f"{phase}_epoch{epoch}_metrics.pt")
        print(f"🐛 DEBUG: Saving metrics to: {metrics_file}")
        torch.save(metrics, metrics_file)
        
        # Also save individual loss file for backward compatibility
        loss_data = {
            'epoch': epoch,
            'phase': phase,
            'loss': train_loss
        }
        loss_file = os.path.join(results_dir, f"{phase}_epoch{epoch}_loss.pt")
        print(f"🐛 DEBUG: Saving loss to: {loss_file}")
        torch.save(loss_data, loss_file)

    num_ensembles = len(swag_pids)
    other_pids = [pid for pid in swag_pids if pid != particle.pid]

    def generate_bootstrap(seed, n_samples):
        torch.manual_seed(seed)
        return torch.randint(0, n_samples, (n_samples,), dtype=torch.long)

    def build_bootstrap_datasets(n_estimators, data_loader):
        X, Y = [], []
        for inputs, labels in data_loader:
            X.append(inputs)
            Y.append(labels)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        n_samples = len(X)
        bootstraps = [
            generate_bootstrap(42 * i, n_samples) for i in range(1, n_estimators + 1)
        ]
        dataloaders = []
        for indices in bootstraps:
            X_b = torch.index_select(X, 0, indices)
            Y_b = torch.index_select(Y, 0, indices)
            dataset = TensorDataset(X_b, Y_b)
            dataloader = DataLoader(
                dataset, batch_size=data_loader.batch_size, shuffle=True
            )
            dataloaders.append(dataloader)
        return dataloaders

    bootstrap_dataloaders = build_bootstrap_datasets(num_ensembles, dataloader)

    # Save initial weights for all particles (epoch 0)
    save_all_particle_weights(particle, other_pids, f"results/{optimizer_name}/pretrain_weights", 0)

    # Training loop
    tq = tqdm(range(pretrain_epochs))
    for e in tq:
        losses = []
        for i, dataloader in enumerate(bootstrap_dataloaders):
            if i == 0:
                for data, label in dataloader:
                    fut = particle.step(loss_fn, data, label)
                    losses += [fut.wait()]
            else:
                for data, label in dataloader:
                    fut = particle.send(swag_pids[i], "SWAG_STEP", loss_fn, data, label)
                    # losses += [fut.wait()]  # Don't wait for other particles
        
        # Calculate average training loss
        avg_train_loss = torch.mean(torch.tensor(losses)).item()
        
        # Evaluate on validation sets
        val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
        val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
        
        # Save metrics for this epoch
        save_epoch_metrics(e + 1, "pretrain", avg_train_loss, val_acc, val_corrupt_acc)
        
        # Update progress bar
        tq.set_postfix({
            "loss": avg_train_loss,
            "val_acc": f"{val_acc:.3f}",
            "val_corrupt_acc": f"{val_corrupt_acc:.3f}"
        })

        # Save weights after each pretraining epoch for all particles
        save_all_particle_weights(particle, other_pids, f"results/{optimizer_name}/pretrain_weights", e + 1)

    # Initialize SWAG
    [particle.send(pid, "SWAG_SWAG", True, cov_mat_rank) for pid in other_pids]
    _swag_swag(particle, True, cov_mat_rank)

    # Save initial SWAG moments for all particles (epoch 0)
    save_swag_moments(particle, f"results/{optimizer_name}/swag_moments", 0)
    for pid in other_pids:
        fut = particle.send(pid, "SAVE_SWAG_MOMENTS", f"results/{optimizer_name}/swag_moments", 0)
        fut.wait()

    tq = tqdm(range(swag_epochs))
    for e in tq:
        losses = []
        for i, dataloader in enumerate(bootstrap_dataloaders):
            if i == 0:
                for data, label in dataloader:
                    fut = particle.step(loss_fn, data, label)
                    losses += [fut.wait()]
            else:
                for data, label in dataloader:
                    fut = particle.send(swag_pids[i], "SWAG_STEP", loss_fn, data, label)
        
        futs = [
            particle.send(pid, "SWAG_SWAG", False, cov_mat_rank) for pid in other_pids
        ]
        _swag_swag(particle, False, cov_mat_rank)
        [f.wait() for f in futs]
        
        # Calculate average training loss
        avg_train_loss = torch.mean(torch.tensor(losses)).item()
        
        # For SWAG phase, evaluate using SWAG mean
        if particle.pid in particle.state and "mom1" in particle.state[particle.pid]:
            # Temporarily set model to SWAG mean for evaluation
            original_params = [p.clone() for p in particle.module.parameters()]
            for param, mean_param in zip(particle.module.parameters(), particle.state[particle.pid]["mom1"]):
                param.data.copy_(mean_param.data)
            
            # Evaluate with SWAG mean
            val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
            val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
            
            # Restore original parameters
            for param, orig_param in zip(particle.module.parameters(), original_params):
                param.data.copy_(orig_param.data)
        else:
            val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
            val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
        
        # Save metrics for this SWAG epoch
        save_epoch_metrics(e + 1, "swag", avg_train_loss, val_acc, val_corrupt_acc)
        
        # Update progress bar
        tq.set_postfix({
            "loss": avg_train_loss,
            "val_acc": f"{val_acc:.3f}",
            "val_corrupt_acc": f"{val_corrupt_acc:.3f}"
        })

        # Save SWAG moments after each SWAG epoch for all particles
        save_swag_moments(particle, f"results/{optimizer_name}/swag_moments", e + 1)
        for pid in other_pids:
            fut = particle.send(pid, "SAVE_SWAG_MOMENTS", f"results/{optimizer_name}/swag_moments", e + 1)
            fut.wait()


def _mswag_particle(
    particle: Particle,
    dataloader,
    loss_fn: Callable,
    cov_mat_rank: int,
    pretrain_epochs: int,
    swag_epochs: int,
    swag_pids: List[int],
    bootstrap: bool,
    val_dataloader=None,
    val_corrupt_dataloader=None,
    save_metrics: bool = True,
    optimizer_name: str = "unknown",
) -> None:
    """Training function for MSWAG particle.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): Training DataLoader.
        loss_fn (Callable): Loss function.
        cov_mat_rank (int): Maximum rank of low rank plus diagonal covariance matrix
        pretrain_epochs (int): Number of pre-training epochs.
        swag_epochs (int): Number of SWAG epochs.
        swag_pids (list[int]): List of SWAG particle IDs.
        bootstrap (bool): Whether to use bootstrap sampling.
        val_dataloader: Validation DataLoader.
        val_corrupt_dataloader: Corrupted validation DataLoader.
        save_metrics (bool): Whether to save metrics at each epoch.
        optimizer_name (str): Name of the optimizer for organizing results.
    """
    
    print(f"🐛 DEBUG: _mswag_particle called with bootstrap={bootstrap}, save_metrics={save_metrics}, optimizer_name={optimizer_name}")
    print(f"🐛 DEBUG: pretrain_epochs={pretrain_epochs}, swag_epochs={swag_epochs}")
    
    def evaluate_model(model, dataloader, device):
        """Evaluate model on a dataloader and return accuracy."""
        if dataloader is None:
            return 0.0
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        model.train()
        return correct / total if total > 0 else 0.0
    
    def save_epoch_metrics(epoch, phase, train_loss, val_acc=None, val_corrupt_acc=None, optimizer_name="unknown"):
        """Save metrics for a given epoch."""
        print(f"🐛 DEBUG: save_epoch_metrics called - epoch={epoch}, phase={phase}, save_metrics={save_metrics}, optimizer_name={optimizer_name}")
        
        if not save_metrics:
            print(f"🐛 DEBUG: save_metrics is False, skipping save")
            return
            
        metrics = {
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_corrupt_acc': val_corrupt_acc
        }
        
        # Create results directory structure
        results_dir = f"results/{optimizer_name}/evaluation_results"
        print(f"🐛 DEBUG: Creating directory: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f"{phase}_epoch{epoch}_metrics.pt")
        print(f"🐛 DEBUG: Saving metrics to: {metrics_file}")
        torch.save(metrics, metrics_file)
        
        # Also save individual loss file for backward compatibility
        loss_data = {
            'epoch': epoch,
            'phase': phase,
            'loss': train_loss
        }
        loss_file = os.path.join(results_dir, f"{phase}_epoch{epoch}_loss.pt")
        print(f"🐛 DEBUG: Saving loss to: {loss_file}")
        torch.save(loss_data, loss_file)

    if bootstrap:
        print(f"🐛 DEBUG: Taking bootstrap path")
        _bootstrap_mswag_particle(
            particle=particle,
            dataloader=dataloader,
            loss_fn=loss_fn,
            cov_mat_rank=cov_mat_rank,
            pretrain_epochs=pretrain_epochs,
            swag_epochs=swag_epochs,
            swag_pids=swag_pids,
            val_dataloader=val_dataloader,
            val_corrupt_dataloader=val_corrupt_dataloader,
            save_metrics=save_metrics,
            optimizer_name=optimizer_name,
        )
    else:
        print(f"🐛 DEBUG: Taking normal training path")
        # TRAINING DONE HERE
        other_pids = [pid for pid in swag_pids if pid != particle.pid]
        
        # Save initial weights before pretraining for all particles (epoch 0)
        save_all_particle_weights(particle, other_pids, f"results/{optimizer_name}/pretrain_weights", 0)

        tq = tqdm(range(pretrain_epochs))
        # Pre-training loop
        for e in tq:
            losses = []
            for data, label in dataloader:
                fut = particle.step(loss_fn, data, label)
                futs = [
                    particle.send(pid, "SWAG_STEP", loss_fn, data, label)
                    for pid in other_pids
                ]
                losses += [fut.wait()]
            
            # Calculate average training loss
            avg_train_loss = torch.mean(torch.tensor(losses)).item()
            
            # Evaluate on validation sets
            val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
            val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
            
            # Save metrics for this epoch
            save_epoch_metrics(e + 1, "pretrain", avg_train_loss, val_acc, val_corrupt_acc, optimizer_name)
            
            # Update progress bar
            tq.set_postfix({
                "loss": avg_train_loss,
                "val_acc": f"{val_acc:.3f}",
                "val_corrupt_acc": f"{val_corrupt_acc:.3f}"
            })

            # Save weights after each pretraining epoch for all particles (epochs 1 to pretrain_epochs)
            save_all_particle_weights(particle, other_pids, f"results/{optimizer_name}/pretrain_weights", e + 1)

        # Initialize SWAG
        [particle.send(pid, "SWAG_SWAG", True, cov_mat_rank) for pid in other_pids]
        _swag_swag(particle, True, cov_mat_rank)

        # Save initial SWAG moments for all particles (epoch 0)
        save_swag_moments(particle, f"results/{optimizer_name}/swag_moments", 0)
        for pid in other_pids:
            fut = particle.send(pid, "SAVE_SWAG_MOMENTS", f"results/{optimizer_name}/swag_moments", 0)
            fut.wait()

        tq = tqdm(range(swag_epochs))
        # SWAG epochs
        for e in tq:
            losses = []
            for data, label in dataloader:
                # Update
                futs = [
                    particle.send(pid, "SWAG_STEP", loss_fn, data, label)
                    for pid in other_pids
                ]
                fut = particle.step(loss_fn, data, label)
                [f.wait() for f in futs]
                losses += [fut.wait()]
            futs = [
                particle.send(pid, "SWAG_SWAG", False, cov_mat_rank)
                for pid in other_pids
            ]
            _swag_swag(particle, False, cov_mat_rank)
            [f.wait() for f in futs]
            
            # Calculate average training loss
            avg_train_loss = torch.mean(torch.tensor(losses)).item()
            
            # For SWAG phase, we can evaluate using SWAG sampling
            # Set model to mean for evaluation
            if particle.pid in particle.state and "mom1" in particle.state[particle.pid]:
                # Temporarily set model to SWAG mean for evaluation
                original_params = [p.clone() for p in particle.module.parameters()]
                for param, mean_param in zip(particle.module.parameters(), particle.state[particle.pid]["mom1"]):
                    param.data.copy_(mean_param.data)
                
                # Evaluate with SWAG mean
                val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
                val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
                
                # Restore original parameters
                for param, orig_param in zip(particle.module.parameters(), original_params):
                    param.data.copy_(orig_param.data)
            else:
                val_acc = evaluate_model(particle.module, val_dataloader, particle.device)
                val_corrupt_acc = evaluate_model(particle.module, val_corrupt_dataloader, particle.device)
            
            # Save metrics for this SWAG epoch
            save_epoch_metrics(e + 1, "swag", avg_train_loss, val_acc, val_corrupt_acc, optimizer_name)
            
            # Update progress bar
            tq.set_postfix({
                "loss": avg_train_loss,
                "val_acc": f"{val_acc:.3f}",
                "val_corrupt_acc": f"{val_corrupt_acc:.3f}"
            })

            # Save SWAG moments after each SWAG epoch for all particles (epochs 1 to swag_epochs)
            save_swag_moments(particle, f"results/{optimizer_name}/swag_moments", e + 1)
            for pid in other_pids:
                fut = particle.send(pid, "SAVE_SWAG_MOMENTS", f"results/{optimizer_name}/swag_moments", e + 1)
                fut.wait()


# =============================================================================
# SWAG Inference
# =============================================================================            fut = self.push_dist.p_launch(0, "LEADER_PRED_DL", data, scale, var_clamp, num_samples, mode, len(self.swag_pids), f_reg)
def _leader_pred_dl(
    particle: Particle,
    dataloader: DataLoader,
    scale: float,
    var_clamp: float,
    num_samples: int,
    mode: List[str],
    num_models: int,
    f_reg: bool = True,
) -> dict:
    """
    Generate predictions using the lead particle in a deep ensemble for a DataLoader.

    Args:
        particle (Particle): The lead particle used for generating predictions.
        dataloader (DataLoader): The DataLoader containing input data for which predictions are to be generated.
        f_reg (bool, optional): Flag indicating whether this is a regression task. Set to false for classification tasks.
            Defaults to True. If set to True, the task is treated as a regression task; otherwise, it is treated as a classification task.
        mode (str, optional): The mode for generating predictions.
            Options include "mean" for mean predictions, "median" for median predictions,
            "min" for minimum predictions, and "max" for maximum predictions.
            Defaults to "mean".

    Returns:
        torch.Tensor: The ensemble predictions for the input data in the DataLoader.

    Note:
        This function generates predictions using the lead particle in a deep ensemble for each batch in the DataLoader.
        It internally calls the `_leader_pred` function to obtain predictions for each batch, and then concatenates
        the results into a single tensor. The `f_reg` flag determines whether the task is treated as a regression or classification task.
    """
    acc = []
    for data, label in dataloader:
        acc += [
            _leader_pred(
                particle,
                data,
                scale=scale,
                var_clamp=var_clamp,
                num_samples=num_samples,
                mode=mode,
                num_models=num_models,
                f_reg=f_reg,
            )
        ]
    results_dict = {}
    for mode_val in mode:
        # print(result.keys())
        results_dict[mode_val] = torch.cat([result[mode_val] for result in acc], dim=0)
    return results_dict


def _leader_pred(
    particle: Particle,
    data: torch.Tensor,
    scale: float,
    var_clamp: float,
    num_samples: int,
    mode: List[str],
    num_models: int,
    f_reg: bool = True,
) -> dict:
    """Generate MSWAG predictions using the lead particle in a MSWAG PusH distribution.

    Args:
        particle (Particle): The lead particle in the MSWAG ensemble.
        dataloader (DataLoader): DataLoader for input data.
        scale (float): Scaling factor for MSWAG sampling.
        var_clamp (float): Clamping value for variance in MSWAG sampling.
        num_samples (int): Number of SWAG samples.
        mode (str): Ensemble prediction mode. Options: "mode", "mean".
        num_models (int): Number of models in the ensemble.

    Returns:
        results_dict (dict): Ensemble predictions for each mode. Access
    """
    other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
    preds = []
    # Perform MSWAG sampling for the lead particle
    preds += [_mswag_pred(particle, data, var_clamp, scale, num_samples)]
    for pid in other_particles:
        preds += [
            particle.send(pid, "SWAG_PRED", data, scale, var_clamp, num_samples).wait()
        ]
    preds = [torch.stack(sublist, dim=0) for sublist in preds]
    t_preds = torch.stack(preds, dim=1)
    # t_preds = preds
    # t_preds = [torch.cat(tensor_list, dim=1)for tensor_list in preds]

    # TODO
    # t_preds = torch.stack(preds, dim=1)
    results_dict = {}
    if f_reg:
        valid_modes = ["mean", "median", "min", "max", "std", "pred"]
        for mode_val in mode:
            assert (
                mode_val in valid_modes
            ), f"Mode {mode_val} not supported. Valid modes are {valid_modes}."
        if "std" in mode:
            results_dict["std"] = torch.std(t_preds, dim=1)
        if "mean" in mode:
            results_dict["mean"] = torch.mean(t_preds, dim=1)
        if "median" in mode:
            results_dict["median"] = torch.median(t_preds, dim=1).values
        if "min" in mode:
            results_dict["min"] = torch.min(t_preds, dim=1).values
        if "max" in mode:
            results_dict["max"] = torch.max(t_preds, dim=1).values
        if "pred" in mode:
            results_dict["pred"] = t_preds
    else:
        valid_modes = ["logits", "prob", "mode", "mean", "median", "std"]
        for mode_val in mode:
            assert (
                mode_val in valid_modes
            ), f"Mode {mode_val} not supported. Valid modes are {valid_modes}."

        # t_preds_softmax = [[torch.nn.functional.softmax(tensor, dim=0) for tensor in sublist] for sublist in preds]
        t_preds_softmax = [entry.softmax(dim=1) for entry in t_preds]
        stacked_preds = torch.stack(t_preds_softmax)
        if "logits" in mode:
            results_dict["logits"] = t_preds
            # results_dict["logits"] = preds
        if "prob" in mode:
            results_dict["prob"] = stacked_preds
        if "mean" in mode:
            mean_probs = torch.mean(stacked_preds, dim=1)
            results_dict["mean"] = torch.argmax(mean_probs, dim=1)
        if "mode" in mode:
            cls = [tensor_list.argmax(dim=1) for tensor_list in t_preds_softmax]
            stacked_cls = torch.stack(cls)
            results_dict["mode"] = torch.mode(stacked_cls, dim=1).values
        if "std" in mode:
            results_dict["std"] = torch.std(stacked_preds, dim=1)
    return results_dict


def _mswag_pred(
    particle: Particle,
    data: torch.Tensor,
    scale: float,
    var_clamp: float,
    num_samples: int,
) -> torch.Tensor:
    """MSWAG sample prediction function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of SWAG samples.

    Returns:
        torch.Tensor: Predictions for the input data.
    """
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]
    cov_mat_sqrt_list = [param for param in particle.state[pid]["cov_mat_sqrt"]]

    scale_sqrt = scale**0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute and store prediction for each SWAG sample
    # preds = {i: [] for i in range(num_samples)}
    preds = []
    for i in range(num_samples):
        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)
        # if covariance draw low rank sample

        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

        cov_sample = cov_mat_sqrt.t().matmul(
            cov_mat_sqrt.new_empty(
                (cov_mat_sqrt.size(0),), requires_grad=False
            ).normal_()
        )
        cov_sample /= (num_samples - 1) ** 0.5

        rand_sample = var_sample + cov_sample
        # rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample
        # Forward pass and store predictions
        # pred = [detach_to_cpu(particle.forward(data).wait()) for data, _ in dataloader]
        preds += [detach_to_cpu(particle.forward(data).wait())]

    mean_preds = []
    # Calculate mean predictions over num_samples
    for n in range(len(preds[0])):
        # Accumulate predictions for each sample
        mean_preds_accum = sum(preds[i][n] for i in range(num_samples))
        # Calculate mean for each data point
        mean_preds.append(mean_preds_accum / num_samples)

    return mean_preds


def _mswag_sample_entry(
    particle: Particle,
    dataloader: DataLoader,
    loss_fn: Callable,
    scale: float,
    var_clamp: float,
    num_samples: int,
    num_models,
) -> None:
    """SWAG sampling entry function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
        num_models (int): Number of models.
    """
    # Unpack state
    state = particle.state
    pid = particle.pid
    other_pids = list(range(1, num_models))

    # Perform SWAG sampling for every particle
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [
        particle.send(
            pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples
        )
        for pid in other_pids
    ]
    ans = [f.wait() for f in futs]

    # Majority vote of majority prediction
    classes = {k: [0 for i in range(10)] for k in range(10)}
    max_preds = [my_ans["max_preds"]] + [
        ans[i - 1]["max_preds"] for i in range(1, num_models)
    ]
    for idx, (data, label) in enumerate(dataloader):
        max_pred = torch.mode(
            torch.stack([max_preds[i][idx] for i in range(num_models)]), dim=0
        ).values
        for x, y in zip(max_pred, label):
            classes[y.item()][x.item()] += 1

    # Majority vote across all SWAG samples
    all_preds = [my_ans["preds"]] + [ans[i - 1]["preds"] for i in range(1, num_models)]
    classes2 = {k: [0 for i in range(10)] for k in range(10)}
    for idx, (data, label) in enumerate(dataloader):
        preds = []
        for m in range(num_models):
            preds += [torch.stack([all_preds[m][i][idx] for i in range(num_samples)])]
        max_pred = torch.mode(torch.cat(preds, dim=0), dim=0).values
        for x, y in zip(max_pred, label):
            classes2[y.item()][x.item()] += 1

    return classes, classes2


def _mswag_sample(
    particle: Particle,
    dataloader: DataLoader,
    loss_fn: Callable,
    scale: float,
    var_clamp: float,
    num_samples: int,
) -> None:
    """SWAG sampling function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
    """
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale**0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    classes = {k: [0 for i in range(10)] for k in range(10)}

    losses = []
    for data, label in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, label)
        cls = pred.softmax(dim=1).argmax(dim=1)
        for x, y in zip(cls, label):
            classes[y.item()][x.item()] += 1
        losses += [loss.detach().to("cpu")]

    # Compute and store prediction for each SWAG sample
    preds = {i: [] for i in range(num_samples)}
    swag_losses = {}
    for i in range(num_samples):
        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        swag_losses = []
        for data, label in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            cls = pred.softmax(dim=1).argmax(dim=1)
            preds[i] += [cls]
            loss = loss_fn(pred, label)
            swag_losses += [loss.detach().to("cpu")]

    # Majority prediction for the current SWAG particle
    max_preds = []
    for n in range(len(preds[0])):
        max_preds += [
            torch.mode(
                torch.stack([preds[i][n] for i in range(num_samples)]), dim=0
            ).values
        ]

    return {
        "classes": classes,
        "losses": torch.mean(torch.tensor(losses)),
        "preds": preds,
        "max_preds": max_preds,
    }


def _mswag_sample_entry_regression(
    particle: Particle,
    dataloader: DataLoader,
    loss_fn: Callable,
    scale: float,
    var_clamp: float,
    num_samples: int,
    num_models,
) -> None:
    """Regression version of SWAG sample entry function.

    Args:
        particle (Particle): MSWAG particle.
        dataloader (DataLoader): DataLoader.
        loss_fn (Callable): Loss function.
        scale (float): Scaling factor.
        var_clamp (float): Variance clamping factor.
        num_samples (int): Number of samples.
        num_models (int): Number of models.
    """
    # Unpack state
    state = particle.state
    pid = particle.pid
    other_pids = list(range(1, num_models))

    # Perform SWAG sampling for every particle
    my_ans = _mswag_sample(particle, dataloader, loss_fn, scale, var_clamp, num_samples)
    futs = [
        particle.send(
            pid, "SWAG_SAMPLE", dataloader, loss_fn, scale, var_clamp, num_samples
        )
        for pid in other_pids
    ]
    ans = [f.wait() for f in futs]

    # Mean prediction across all models
    mean_preds = torch.mean(
        torch.stack(
            [my_ans["mean_preds"]]
            + [ans[i - 1]["mean_preds"] for i in range(1, num_models)]
        ),
        dim=0,
    )

    return mean_preds


def _mswag_sample_regression(
    particle: Particle,
    dataloader: DataLoader,
    loss_fn: Callable,
    scale: float,
    var_clamp: float,
    num_samples: int,
) -> None:
    """
    Modified SWAG sample function for regression problems.

    Args:
        particle (Particle): The SWAG particle.
        dataloader (DataLoader): DataLoader containing the data.
        loss_fn (Callable): Loss function used for computing losses.
        scale (float): Scaling factor for the SWAG sample.
        var_clamp (float): Clamping value for the variance.
        num_samples (int): Number of SWAG samples to generate.

    Returns:
        dict: Dictionary containing computed losses and predictions.

    """
    state = particle.state
    pid = particle.pid
    # Gather
    mean_list = [param for param in particle.state[pid]["mom1"]]
    sq_mean_list = [param for param in particle.state[pid]["mom2"]]

    scale_sqrt = scale**0.5
    mean = flatten(mean_list)
    sq_mean = flatten(sq_mean_list)

    # Compute original loss
    losses = []
    for data, target in tqdm(dataloader):
        pred = detach_to_cpu(particle.forward(data).wait())
        loss = loss_fn(pred, target)
        losses += [loss.detach().to("cpu")]

    # Compute and store prediction for each SWAG sample
    preds = {i: [] for i in range(num_samples)}
    swag_losses = {}
    for i in range(num_samples):
        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Update
        samples_list = unflatten_like(sample, mean_list)

        for param, sample in zip(particle.module.parameters(), samples_list):
            param.data = sample

        swag_losses = []
        for data, target in tqdm(dataloader):
            pred = detach_to_cpu(particle.forward(data).wait())
            preds[i] += [pred]
            loss = loss_fn(pred, target)
            swag_losses += [loss.detach().to("cpu")]

    # Mean prediction for the current SWAG particle
    mean_preds = torch.mean(
        torch.stack([torch.stack(v) for v in preds.values()]), dim=0
    )

    return {
        "losses": torch.mean(torch.tensor(losses)),
        "mean_preds": mean_preds,
        "swag_losses": swag_losses,
    }


class MultiSWAG(Infer):
    """
    MultiSWAG class for running MultiSWAG models.

    Args:
        mk_nn (Callable): The base model to be ensembled.
        *args: Any arguments required for the base model initialization.
        num_devices (int): The desired number of GPU devices to utilize.
        cache_size (int): The size of the cache used to store particles.
        view_size (int): The number of particles to consider storing in the cache.

    """

    def __init__(
        self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4
    ) -> None:
        """
        Initialize the MultiSWAG model.

        Args:
            mk_nn (Callable): The base model to be ensembled.
            *args: Any arguments required for the base model initialization.
            num_devices (int): The desired number of GPU devices to utilize.
            cache_size (int): The size of the cache used to store particles.
            view_size (int): The number of particles to consider storing in the cache.

        """
        super(MultiSWAG, self).__init__(
            mk_nn,
            *args,
            num_devices=num_devices,
            cache_size=cache_size,
            view_size=view_size,
        )
        self.swag_pids = []
        self.sample_pids = []

    def bayes_infer(
        self,
        dataloader: DataLoader,
        pretrain_epochs: int,
        swag_epochs: int,
        create_optimizer: Callable,
        loss_fn: Callable = torch.nn.MSELoss(),
        lr: float = 0.01,
        num_models: int = 1,
        cov_mat_rank: int = 20,
        prior=False,
        random_seed=False,
        bootstrap=False,
        mswag_entry=_mswag_particle,
        mswag_state={},
        f_save=False,
        mswag_sample_entry=_mswag_sample_entry,
        mswag_sample=_mswag_sample,
        val_dataloader=None,
        val_corrupt_dataloader=None,
        save_metrics: bool = True,
        optimizer_name: str = "unknown",
    ):
        """
        Perform Bayesian inference using MultiSWAG.

        Args:
            dataloader (DataLoader): DataLoader containing the data.
            loss_fn (Callable): Loss function used for training.
            num_models (int): Number of models to be ensembled.
            cov_mat_rank (int): Maximum rank of low rank plus diagonal covariance matrix
            lr (float): Learning rate for training.
            pretrain_epochs (int): Number of epochs for pretraining.
            swag_epochs (int): Number of epochs for SWAG training.
            mswag_entry (Callable): Training loop for deep ensemble.
            mswag_state (dict): State variables for ensembled models.
            f_save (bool): Flag to save each particle/model.
            mswag_sample_entry (Callable): Sampling function.
            mswag_sample (Callable): MultiSWAG sample function.
            val_dataloader: Validation DataLoader.
            val_corrupt_dataloader: Corrupted validation DataLoader.
            save_metrics (bool): Whether to save metrics at each epoch.
            optimizer_name (str): Name of the optimizer for organizing results.

        Returns:
            None

        """
        if "n" in mswag_state:
            raise ValueError(
                f"Cannot run with state {mswag_state['n']}. Please rename."
            )
        mswag_state["n"] = 1

        mk_optim = create_optimizer(lr)

        if random_seed:
            train_keys = torch.randint(
                0, int(1e9), (num_models,), dtype=torch.int64
            ).tolist()
        else:
            train_keys = [-1] * num_models

        def mk_swag(model_num):
            # Particle for parameter
            if model_num == 0:
                param_pid = self.push_dist.p_create(
                    mk_optim,
                    mk_scheduler,
                    prior,
                    train_keys[0],
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_PARTICLE": mswag_entry,
                        "SWAG_SAMPLE_ENTRY": mswag_sample_entry,
                        "LEADER_PRED": _leader_pred,
                        "LEADER_PRED_DL": _leader_pred_dl,
                    },
                    state=mswag_state,
                )
            else:
                param_pid = self.push_dist.p_create(
                    mk_optim,
                    mk_scheduler,
                    prior,
                    train_keys[model_num],
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_STEP": _swag_step,
                        "SWAG_SWAG": _swag_swag,
                        "SWAG_SAMPLE": mswag_sample,
                        "SWAG_PRED": _mswag_pred,
                        "GET_MODULE_STATE": lambda p: p.module.state_dict(),
                        "SAVE_SWAG_MOMENTS": lambda p, save_dir, epoch: save_swag_moments(
                            p, save_dir, epoch
                        ),
                    },
                    state=mswag_state,
                )
                # Add optimizer name to particle for saving purposes
                # self.push_dist.particles[param_pid].optimizer_name = optimizer_name
            return param_pid

        self.swag_pids = [mk_swag(n) for n in range(num_models)]
        for pid in self.swag_pids:
            if pid in mswag_state:
                raise ValueError(f"Cannot run with state {pid}. Please rename.")

        self.push_dist.p_wait(
            [
                self.push_dist.p_launch(
                    self.swag_pids[0],
                    "SWAG_PARTICLE",
                    dataloader,
                    loss_fn,
                    cov_mat_rank,
                    pretrain_epochs,
                    swag_epochs,
                    self.swag_pids,
                    bootstrap,
                    val_dataloader,
                    val_corrupt_dataloader,
                    save_metrics,
                    optimizer_name,
                )
            ]
        )

        if f_save:
            self.push_dist.save()

    def posterior_pred(
        self,
        data: DataLoader,
        loss_fn=torch.nn.MSELoss(),
        num_samples: int = 20,
        scale: float = 1.0,
        var_clamp: float = 1e-30,
        mode: List[str] = ["mean"],
        f_reg: bool = True,
    ):
        """
        Generate posterior predictions using MultiSWAG.

        Args:
            dataloader (DataLoader): DataLoader containing the data.
            loss_fn (Callable): Loss function used for computing losses.
            num_samples (int): Number of samples to generate.
            scale (float): Scaling factor for the SWAG sample.
            var_clamp (float): Clamping value for the variance.

        Returns:
            None

        """

        if isinstance(data, torch.Tensor):
            fut = self.push_dist.p_launch(
                0,
                "LEADER_PRED",
                data,
                scale,
                var_clamp,
                num_samples,
                mode,
                len(self.swag_pids),
                f_reg,
            )
            return self.push_dist.p_wait([fut])[fut._fid]
        elif isinstance(data, DataLoader):
            fut = self.push_dist.p_launch(
                0,
                "LEADER_PRED_DL",
                data,
                scale,
                var_clamp,
                num_samples,
                mode,
                len(self.swag_pids),
                f_reg,
            )
            return self.push_dist.p_wait([fut])[fut._fid]
        else:
            raise ValueError(f"Data of type {type(data)} not supported ...")

    def load_from_pretrained_epoch(
        self,
        opt: str,
        epoch: int,
        num_models: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Load a MultiSWAG ensemble from pretrained weights at a specific epoch."""
        weights_dir = os.path.join("results", opt, "pretrain_weights")
        
        # Create particles for each model
        self.swag_pids = []
        for model_num in range(num_models):
            # Load pretrained weights
            weights_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_weights.pt")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
            
            # Load weights
            state_dict = torch.load(weights_path, map_location=device)
            
            # Create particle for this model
            if model_num == 0:
                # Lead particle needs prediction handlers
                param_pid = self.push_dist.p_create(
                    lambda params: torch.optim.Adam(params, lr=0.01),
                    mk_scheduler,
                    False,  # no prior
                    -1,  # no random seed
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_STEP": _swag_step,
                        "SWAG_SWAG": _swag_swag,
                        "SWAG_SAMPLE": _mswag_sample,
                        "SWAG_PRED": _mswag_pred,
                        "GET_MODULE_STATE": lambda p: p.module.state_dict(),
                        "SET_MODULE_STATE": lambda p, state_dict: p.module.load_state_dict(state_dict),
                        "SET_SWAG_STATE": lambda p, mom1, mom2, cov_mat_sqrt: _set_swag_state_helper(p, mom1, mom2, cov_mat_sqrt),
                        "SET_TO_MEAN": lambda p: [param.data.copy_(mean_param.data) 
                                                for param, mean_param in zip(p.module.parameters(), p.state[p.pid]['mom1'])],
                        "LEADER_PRED": _leader_pred,
                        "LEADER_PRED_DL": _leader_pred_dl,
                    },
                    state={"n": 1},  # Initialize with n=1 since we're starting SWAG from this pretrained point
                )
            else:
                param_pid = self.push_dist.p_create(
                    lambda params: torch.optim.Adam(params, lr=0.01),
                    mk_scheduler,
                    False,  # no prior
                    -1,  # no random seed
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_STEP": _swag_step,
                        "SWAG_SWAG": _swag_swag,
                        "SWAG_SAMPLE": _mswag_sample,
                        "SWAG_PRED": _mswag_pred,
                        "GET_MODULE_STATE": lambda p: p.module.state_dict(),
                        "SET_MODULE_STATE": lambda p, state_dict: p.module.load_state_dict(state_dict),
                        "SET_SWAG_STATE": lambda p, mom1, mom2, cov_mat_sqrt: _set_swag_state_helper(p, mom1, mom2, cov_mat_sqrt),
                        "SET_TO_MEAN": lambda p: [param.data.copy_(mean_param.data) 
                                                for param, mean_param in zip(p.module.parameters(), p.state[p.pid]['mom1'])],
                    },
                    state={"n": 1},  # Initialize with n=1 since we're starting SWAG from this pretrained point
                )
            self.swag_pids.append(param_pid)
            
            # Load weights into particle using message passing
            fut = self.push_dist.p_launch(param_pid, "SET_MODULE_STATE", state_dict)
            self.push_dist.p_wait([fut])
            
            # Initialize SWAG moments after loading weights (this creates mom1, mom2, cov_mat_sqrt from current model state)
            fut = self.push_dist.p_launch(param_pid, "SWAG_SWAG", True, 20)
            self.push_dist.p_wait([fut])

    def load_from_swag_epoch(
        self,
        opt: str,
        epoch: int,
        num_models: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Load a MultiSWAG ensemble from SWAG moments at a specific epoch."""
        weights_dir = os.path.join("results", opt, "swag_moments")
        
        # Create particles for each model
        self.swag_pids = []
        for model_num in range(num_models):
            # Load SWAG moments using model_num to match the saving pattern
            mom1_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_mom1.pt")
            mom2_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_mom2.pt")
            cov_path = os.path.join(weights_dir, f"particle{model_num}_epoch{epoch}_cov_mat_sqrt.pt")
            
            if not all(os.path.exists(p) for p in [mom1_path, mom2_path, cov_path]):
                raise FileNotFoundError(f"SWAG moments not found in {weights_dir}")
            
            # Load moments
            mom1 = torch.load(mom1_path, map_location=device)
            mom2 = torch.load(mom2_path, map_location=device)
            cov_mat_sqrt = torch.load(cov_path, map_location=device)
            
            # Create particle for this model
            if model_num == 0:
                # Lead particle needs prediction handlers
                param_pid = self.push_dist.p_create(
                    lambda params: torch.optim.Adam(params, lr=0.01),
                    mk_scheduler,
                    False,
                    -1,
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_STEP": _swag_step,
                        "SWAG_SWAG": _swag_swag,
                        "SWAG_SAMPLE": _mswag_sample,
                        "SWAG_PRED": _mswag_pred,
                        "GET_MODULE_STATE": lambda p: p.module.state_dict(),
                        "SET_MODULE_STATE": lambda p, state_dict: p.module.load_state_dict(state_dict),
                        "SET_SWAG_STATE": lambda p, mom1, mom2, cov_mat_sqrt: _set_swag_state_helper(p, mom1, mom2, cov_mat_sqrt),
                        "SET_TO_MEAN": lambda p: [param.data.copy_(mean_param.data) 
                                                for param, mean_param in zip(p.module.parameters(), p.state[p.pid]['mom1'])],
                        "LEADER_PRED": _leader_pred,
                        "LEADER_PRED_DL": _leader_pred_dl,
                    },
                    state={"n": epoch},
                )
            else:
                param_pid = self.push_dist.p_create(
                    lambda params: torch.optim.Adam(params, lr=0.01),
                    mk_scheduler,
                    False,
                    -1,
                    device=(model_num % self.num_devices),
                    receive={
                        "SWAG_STEP": _swag_step,
                        "SWAG_SWAG": _swag_swag,
                        "SWAG_SAMPLE": _mswag_sample,
                        "SWAG_PRED": _mswag_pred,
                        "GET_MODULE_STATE": lambda p: p.module.state_dict(),
                        "SET_MODULE_STATE": lambda p, state_dict: p.module.load_state_dict(state_dict),
                        "SET_SWAG_STATE": lambda p, mom1, mom2, cov_mat_sqrt: _set_swag_state_helper(p, mom1, mom2, cov_mat_sqrt),
                        "SET_TO_MEAN": lambda p: [param.data.copy_(mean_param.data) 
                                                for param, mean_param in zip(p.module.parameters(), p.state[p.pid]['mom1'])],
                    },
                    state={"n": epoch},
                )
            self.swag_pids.append(param_pid)
            
            # Initialize SWAG state structure
            fut = self.push_dist.p_launch(param_pid, "SWAG_SWAG", True, 20)
            self.push_dist.p_wait([fut])
            
            # Set the loaded moments and model weights to mean
            fut = self.push_dist.p_launch(param_pid, "SET_SWAG_STATE", mom1, mom2, cov_mat_sqrt)
            self.push_dist.p_wait([fut])


# =============================================================================
# Multi-Swag Training
# =============================================================================


def train_mswag(
    dataloader: DataLoader,
    loss_fn: Callable,
    create_optimizer: Callable,
    pretrain_epochs: int,
    swag_epochs: int,
    nn: Callable,
    *args,
    lr: float = 0.01,
    num_devices=1,
    cache_size: int = 4,
    view_size: int = 4,
    num_models: int = 1,
    cov_mat_rank: int = 20,
    prior=False,
    random_seed=False,
    bootstrap=False,
    mswag_entry=_mswag_particle,
    mswag_state={},
    f_save=False,
    mswag_sample_entry=_mswag_sample_entry,
    mswag_sample=_mswag_sample,
    val_dataloader=None,
    val_corrupt_dataloader=None,
    save_metrics: bool = True,
    optimizer_name: str = "unknown",
):
    """
    Train a MultiSWAG model.

    Args:
        dataloader (DataLoader): DataLoader containing the training data.
        loss_fn (Callable): Loss function used for training.
        pretrain_epochs (int): Number of epochs for pretraining.
        swag_epochs (int): Number of epochs for SWAG training.
        num_models (int): Number of models to use in MultiSWAG.
        cov_mat_rank (int): Maximum rank of low rank plus diagonal covariance matrix
        cache_size (int): Size of the cache for MultiSWAG.
        view_size (int): Size of the view for MultiSWAG.
        nn (Callable): Callable function representing the neural network model.
        *args: Additional arguments for the neural network.
        num_devices (int): Number of devices for training (default is 1).
        lr (float): Learning rate for training (default is 1e-3).
        mswag_entry (Callable): MultiSWAG entry function (default is _mswag_particle).
        mswag_state (dict): Initial state for MultiSWAG (default is {}).
        f_save (bool): Flag to save the model (default is False).
        mswag_sample_entry (Callable): MultiSWAG sample entry function (default is _mswag_sample_entry).
        mswag_sample (Callable): MultiSWAG sample function (default is _mswag_sample).
        val_dataloader: Validation DataLoader.
        val_corrupt_dataloader: Corrupted validation DataLoader.
        save_metrics (bool): Whether to save metrics at each epoch.
        optimizer_name (str): Name of the optimizer for organizing results.

    Returns:
        MultiSWAG: Trained MultiSWAG model.

    """
    mswag = MultiSWAG(
        nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size
    )
    mswag.bayes_infer(
        dataloader,
        pretrain_epochs=pretrain_epochs,
        swag_epochs=swag_epochs,
        loss_fn=loss_fn,
        create_optimizer=create_optimizer,
        lr=lr,
        num_models=num_models,
        cov_mat_rank=cov_mat_rank,
        prior=prior,
        random_seed=random_seed,
        bootstrap=bootstrap,
        mswag_entry=mswag_entry,
        mswag_state=mswag_state,
        f_save=f_save,
        mswag_sample_entry=mswag_sample_entry,
        mswag_sample=mswag_sample,
        val_dataloader=val_dataloader,
        val_corrupt_dataloader=val_corrupt_dataloader,
        save_metrics=save_metrics,
        optimizer_name=optimizer_name,
    )
    return mswag
