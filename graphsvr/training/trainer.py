"""High-level zero-shot optimizer for GraphSVR."""

from __future__ import annotations

import datetime as dt
import pprint
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graphsvr.data.dataset import GraphSVRDataset, collate_keep_single
from graphsvr.geom.rigid_transforms import wrap_3d_image_torch_batch
from graphsvr.models.graph_svr import GraphSVR
from graphsvr.training.losses import LocalMutualInformation, MultiScaleRegistrationLoss, NGFLoss


class MinMaxNormalize:
    """Min-max normalize one tensor to a configurable range."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-8):
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val.")
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        t_min = tensor.amin()
        t_max = tensor.amax()
        scale = (t_max - t_min).clamp_min(self.eps)
        normalized = (tensor - t_min) / scale
        return normalized * (self.max_val - self.min_val) + self.min_val


class GraphSVRTrainer:
    """Optimize GraphSVR for a single case and save the estimated stack transforms."""

    SUPPORTED_LOSSES = {"LMI", "Grad"}

    def __init__(
        self,
        path_to_case: str,
        rigid_config: Dict[str, Any],
        GNN_args: Dict[str, Any],
        lr: float = 1e-4,
        batch_size: int = 20,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        tensorboard: bool = True,
        save_exp_path: Optional[str] = None,
        multi_scale_loss: bool = False,
        multi_scale_loss_scales: int = 3,
        k_for_a_matrix: int = 8,
        node_features_size: int = 64,
        loss_weights: Optional[Dict[str, float]] = None,
        LMI_patch_size: int = 5,
        edges_components: Optional[Dict[str, Any]] = None,
        exp_name: str = "zero_shot_SVR",
        min_iter: int = 50,
        early_stopping: Optional[int] = 20,
        device: str = "cpu",
        num_workers: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if early_stopping is not None and early_stopping <= 0:
            raise ValueError("early_stopping must be positive or None.")

        loss_weights = {"LMI": 1.0, "Grad": 1.0} if loss_weights is None else dict(loss_weights)
        unknown_losses = set(loss_weights) - self.SUPPORTED_LOSSES
        if unknown_losses:
            raise ValueError(f"Unsupported loss(es): {sorted(unknown_losses)}")
        if not loss_weights or any(weight < 0 for weight in loss_weights.values()):
            raise ValueError("At least one non-negative registration loss weight is required.")

        edges_components = (
            {"use_q_space": True, "use_timing": True, "use_x_space": True}
            if edges_components is None
            else dict(edges_components)
        )

        self.device = torch.device(device)
        self.edges_components = edges_components
        self.tensorboard = tensorboard
        self.writer: Optional[SummaryWriter] = None
        if tensorboard:
            self.writer = SummaryWriter(comment=exp_name, flush_secs=5)
            self.writer.add_text("Config/rigid", pprint.pformat(rigid_config))
            self.writer.add_text("Config/GNN", pprint.pformat(GNN_args))
            self.writer.add_text("Config/loss_weights", pprint.pformat(loss_weights))

        normalize = MinMaxNormalize()
        self.dataset = GraphSVRDataset(path_to_case, transform=normalize)
        self.dl = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=False,
            collate_fn=collate_keep_single,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        if len(self.dl) == 0:
            raise ValueError("The case produced no slice-group samples.")

        first_sample = self.dataset[0]
        n_slices_per_stack = first_sample[-2].shape[0]
        self.vox_size = self.dataset.voxel_size
        self.path_to_case = str(path_to_case)
        self.n_b0 = int((self.dataset.bvals == 0).sum().item())
        self.num_of_directions = int((self.dataset.bvals != 0).sum().item())
        self.mb_factor = int(n_slices_per_stack)
        self.dt = self.dataset.dt
        self.dwi_affine = self.dataset.affine

        rigid_config = dict(rigid_config)
        rigid_config["voxel_size"] = self.vox_size
        rigid_config["image_size"] = self.dataset.mask.shape

        self.model = GraphSVR(
            rigid_config=rigid_config,
            GNN_args=dict(GNN_args),
            features_size=node_features_size,
            k_for_a_matrix=k_for_a_matrix,
            device=str(self.device),
        ).to(self.device)
        self.model.train()

        def count_parameters(module: torch.nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        print(f"StackEncoder parameters: {count_parameters(self.model.StackEncoder):,}")
        print(f"VolumeEncoder parameters: {count_parameters(self.model.VolumeEncoder):,}")
        print(f"GNN model parameters: {count_parameters(self.model.model):,}")

        self.loss_weights = loss_weights
        self.reg_losses: Dict[str, torch.nn.Module] = {}
        for loss_name in loss_weights:
            if loss_name == "LMI":
                self.reg_losses[loss_name] = LocalMutualInformation(
                    patch_size=LMI_patch_size, reduction="mean"
                ).to(self.device)
            elif loss_name == "Grad":
                self.reg_losses[loss_name] = NGFLoss(
                    spacing=self.vox_size, reduction="mean"
                ).to(self.device)

        if multi_scale_loss:
            self.reg_losses = {
                name: MultiScaleRegistrationLoss(
                    loss_func=loss_func,
                    scales=multi_scale_loss_scales,
                    kernel=3,
                    num_dim=2,
                ).to(self.device)
                for name, loss_func in self.reg_losses.items()
            }

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.min_iter = min_iter
        self.early_stopping = early_stopping

        self.exp_dir_path: Optional[Path]
        if save_exp_path is not None:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir_path = Path(save_exp_path) / f"{exp_name}_{timestamp}"
            self.exp_dir_path.mkdir(parents=True, exist_ok=False)
        else:
            self.exp_dir_path = None

    def forward(self, data):
        return self.forward_only_reg_stacks(data)

    def fit(self, num_epochs: int) -> Path:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        epochs_without_improvement = 0
        stopped_early = False

        num_batches = len(self.dl)
        with tqdm.tqdm(
            desc="slice-to-volume zero-shot registration",
            total=num_epochs * num_batches,
        ) as pbar:
            for epoch in range(num_epochs):
                self.model.train()
                epoch_loss_sum = 0.0
                epoch_sub_loss_sums: Dict[str, float] = {}

                for data in self.dl:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses_dict, _ = self.forward(data)
                    loss = self.calc_loss(losses_dict)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss encountered at epoch {epoch}: {loss.detach().cpu().item()}"
                        )
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)

                    epoch_loss_sum += float(loss.detach().item())
                    for key, val in losses_dict.items():
                        epoch_sub_loss_sums[key] = epoch_sub_loss_sums.get(key, 0.0) + float(
                            val.detach().item()
                        )

                epoch_mean_loss = epoch_loss_sum / num_batches
                self.scheduler.step(epoch_mean_loss)

                if epoch_mean_loss < best_loss:
                    best_loss = epoch_mean_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if self.writer is not None:
                    self.writer.add_scalar(
                        "learning_rate", self.optimizer.param_groups[0]["lr"], epoch
                    )
                    self.writer.add_scalar("loss_epoch_mean", epoch_mean_loss, epoch)
                    for key, sum_val in epoch_sub_loss_sums.items():
                        self.writer.add_scalar(f"{key}_epoch_mean", sum_val / num_batches, epoch)

                if (
                    self.early_stopping is not None
                    and epoch >= self.min_iter
                    and epochs_without_improvement >= self.early_stopping
                ):
                    print(
                        f"Early stopping at epoch {epoch}; best mean loss was {best_loss:.6g}."
                    )
                    stopped_early = True
                    break

        self.model.load_state_dict(best_state)
        transforms = self.predict_all()
        result_path = self._save_results(transforms, best_loss, stopped_early)
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        return result_path

    def _save_results(
        self, transforms: torch.Tensor, best_loss: float, stopped_early: bool
    ) -> Path:
        output_dir = self.exp_dir_path or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "pred_rigid_trans.pt"
        torch.save(transforms, result_path)
        metadata = {
            "best_mean_loss": best_loss,
            "stopped_early": stopped_early,
            "num_stacks": int(transforms.shape[0]),
            "transform_shape": list(transforms.shape),
            "transform_convention": "PyTorch affine_grid theta matrices mapping output normalized coordinates to input normalized coordinates",
        }
        torch.save(metadata, output_dir / "run_metadata.pt")
        return result_path

    @torch.no_grad()
    def predict_all(self) -> torch.Tensor:
        """Run one coherent inference pass using the final/best model state."""
        was_training = self.model.training
        self.model.eval()
        predictions = []
        for data in self.dl:
            dwi_stacks, stacks_bvecs, _, _, t1_image, stacks_indices, timing = data
            predictions.append(
                self.model(
                    dwi_stacks.to(self.device, non_blocking=True),
                    stacks_bvecs.to(self.device, non_blocking=True),
                    stacks_indices.to(self.device, non_blocking=True),
                    t1_image.to(self.device, non_blocking=True),
                    timing.to(self.device, non_blocking=True),
                    self.edges_components,
                    self.dwi_affine,
                ).detach().cpu()
            )
        if was_training:
            self.model.train()
        return torch.cat(predictions, dim=0)

    def calc_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not losses:
            raise ValueError("No registration losses were computed.")
        final_loss = sum(self.loss_weights[key] * value for key, value in losses.items())
        return final_loss.float()

    def forward_only_reg_stacks(self, data):
        dwi_stacks, stacks_bvecs, _, _, t1_image, stacks_indices, timing = data
        t1_image = t1_image.to(self.device, non_blocking=True)
        dwi_stacks = dwi_stacks.to(self.device, non_blocking=True)
        stacks_bvecs = stacks_bvecs.to(self.device, non_blocking=True)
        stacks_indices = stacks_indices.to(self.device, non_blocking=True)
        timing = timing.to(self.device, non_blocking=True)

        rigid_trans = self.model(
            dwi_stacks,
            stacks_bvecs,
            stacks_indices,
            t1_image,
            timing,
            self.edges_components,
            self.dwi_affine,
        )

        t1_repeated = t1_image.unsqueeze(0).expand(dwi_stacks.shape[0], -1, -1, -1)
        t1_warped = wrap_3d_image_torch_batch(rigid_trans, t1_repeated)
        batch_size, height, width, _ = t1_warped.shape
        n_slices = stacks_indices.shape[1]

        idx = stacks_indices.to(dtype=torch.long)
        idx = idx[:, None, None, :].expand(batch_size, height, width, n_slices)
        t1_sampled = torch.gather(t1_warped, dim=3, index=idx)

        b, w, h, s = dwi_stacks.shape
        moving = dwi_stacks.permute(0, 3, 1, 2).reshape(b * s, 1, w, h)
        fixed = t1_sampled.permute(0, 3, 1, 2).reshape(b * s, 1, w, h)
        losses = {
            name: loss_func(moving, fixed) for name, loss_func in self.reg_losses.items()
        }
        return losses, rigid_trans


__all__ = ["GraphSVRTrainer", "MinMaxNormalize"]
