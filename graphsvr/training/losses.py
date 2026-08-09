"""
Registration loss definitions for GraphSVR.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMutualInformation(nn.Module):
    """Local Mutual Information for non-overlapping patches."""

    def __init__(
        self,
        sigma_ratio=1,
        minval: float = 0.0,
        maxval: float = 1.0,
        num_bin: int = 32,
        patch_size: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()

        self.reduction = reduction

        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, num_bin)
        num_bins = len(bin_centers)

        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.register_buffer("vol_bin_centers", vol_bin_centers, persistent=False)
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0.0, self.max_clip)
        y_true = torch.clamp(y_true, 0.0, self.max_clip)

        o = [1, 1, int(np.prod(self.vol_bin_centers.shape))]
        vbc = self.vol_bin_centers.view(o).to(device=y_pred.device, dtype=y_pred.dtype)

        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (
                z_r // 2,
                z_r - z_r // 2,
                y_r // 2,
                y_r - y_r // 2,
                x_r // 2,
                x_r - x_r // 2,
                0,
                0,
                0,
                0,
            )
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (
                y_r // 2,
                y_r - y_r // 2,
                x_r // 2,
                x_r - x_r // 2,
                0,
                0,
                0,
                0,
            )
        else:
            raise Exception(f"Supports 2D and 3D but not {list(y_pred.size())}")

        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        if ndim == 3:
            y_true_patch = torch.reshape(
                y_true,
                (
                    y_true.shape[0],
                    y_true.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                    (z + z_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = y_true_patch.reshape(-1, self.patch_size**3, 1)

            y_pred_patch = torch.reshape(
                y_pred,
                (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                    (z + z_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = y_pred_patch.reshape(-1, self.patch_size**3, 1)
        else:
            y_true_patch = torch.reshape(
                y_true,
                (
                    y_true.shape[0],
                    y_true.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = y_true_patch.reshape(-1, self.patch_size**2, 1)

            y_pred_patch = torch.reshape(
                y_pred,
                (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = y_pred_patch.reshape(-1, self.patch_size**2, 1)

        I_a_patch = torch.exp(-self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(-self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / (self.patch_size**ndim)

        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6

        mi = torch.sum(torch.sum(pab * torch.log(pab / (papb + 1e-6) + 1e-6), dim=1), dim=1)
        return mi

    def forward(self, y_true, y_pred):
        if len(y_pred.shape) == 3:
            mi = self.local_mi(y_true.unsqueeze(1), y_pred.unsqueeze(1))
        else:
            mi = self.local_mi(y_true, y_pred)

        loss = -mi

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction in ("none", None):
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


class NGFLoss(nn.Module):
    """
    Normalized gradient fields loss for 2D/3D images.
    """

    def __init__(self, eps: float = 1e-6, spacing=None, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.spacing = spacing

        kx2 = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        ) / 8.0
        ky2 = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ) / 8.0
        self.register_buffer("kx2", kx2.view(1, 1, 3, 3))
        self.register_buffer("ky2", ky2.view(1, 1, 3, 3))

        d = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
        s = torch.tensor([1.0, 2.0, 1.0]) / 4.0

        k_h = s[:, None, None] * d[None, :, None] * s[None, None, :]
        k_w = s[:, None, None] * s[None, :, None] * d[None, None, :]
        k_d = d[:, None, None] * s[None, :, None] * s[None, None, :]

        self.register_buffer("k_dx", k_h.view(1, 1, 3, 3, 3))
        self.register_buffer("k_dy", k_w.view(1, 1, 3, 3, 3))
        self.register_buffer("k_dz", k_d.view(1, 1, 3, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")
        if pred.dim() not in (4, 5):
            raise ValueError(f"Expected 2D [B,1,X,Y] or 3D [B,1,X,Y,Z], got {pred.shape}")

        if pred.dim() == 4:
            gx_p = F.conv2d(pred, self.kx2.to(dtype=pred.dtype), padding=1)
            gy_p = F.conv2d(pred, self.ky2.to(dtype=pred.dtype), padding=1)
            gx_t = F.conv2d(target, self.kx2.to(dtype=target.dtype), padding=1)
            gy_t = F.conv2d(target, self.ky2.to(dtype=target.dtype), padding=1)

            spacing_xy = self.spacing[:-1] if self.spacing is not None else None
            if spacing_xy is None:
                sx, sy = 1.0, 1.0
            else:
                sx, sy = map(float, spacing_xy)

            gx_p = gx_p / sx
            gy_p = gy_p / sy
            gx_t = gx_t / sx
            gy_t = gy_t / sy

            n_p = torch.sqrt(gx_p**2 + gy_p**2 + self.eps)
            n_t = torch.sqrt(gx_t**2 + gy_t**2 + self.eps)

            gx_p, gy_p = gx_p / n_p, gy_p / n_p
            gx_t, gy_t = gx_t / n_t, gy_t / n_t

            dot = gx_p * gx_t + gy_p * gy_t
        else:
            if self.spacing is None:
                sx, sy, sz = 1.0, 1.0, 1.0
            else:
                sx, sy, sz = map(float, self.spacing)

            pred_p = pred.permute(0, 1, 4, 2, 3)
            targ_p = target.permute(0, 1, 4, 2, 3)

            gX_p = F.conv3d(pred_p, self.k_dx.to(dtype=pred.dtype), padding=1)
            gY_p = F.conv3d(pred_p, self.k_dy.to(dtype=pred.dtype), padding=1)
            gZ_p = F.conv3d(pred_p, self.k_dz.to(dtype=pred.dtype), padding=1)

            gX_t = F.conv3d(targ_p, self.k_dx.to(dtype=target.dtype), padding=1)
            gY_t = F.conv3d(targ_p, self.k_dy.to(dtype=target.dtype), padding=1)
            gZ_t = F.conv3d(targ_p, self.k_dz.to(dtype=target.dtype), padding=1)

            gX_p = gX_p / sx
            gY_p = gY_p / sy
            gZ_p = gZ_p / sz
            gX_t = gX_t / sx
            gY_t = gY_t / sy
            gZ_t = gZ_t / sz

            n_p = torch.sqrt(gX_p**2 + gY_p**2 + gZ_p**2 + self.eps)
            n_t = torch.sqrt(gX_t**2 + gY_t**2 + gZ_t**2 + self.eps)

            gX_p, gY_p, gZ_p = gX_p / n_p, gY_p / n_p, gZ_p / n_p
            gX_t, gY_t, gZ_t = gX_t / n_t, gY_t / n_t, gZ_t / n_t

            dot = gX_p * gX_t + gY_p * gY_t + gZ_p * gZ_t
            dot = dot.permute(0, 1, 3, 4, 2)

        loss = 1.0 - dot**2

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiScaleRegistrationLoss(nn.Module):
    def __init__(self, loss_func, scales=3, kernel=3, num_dim=2, weights=None):
        super().__init__()
        self.loss_func = loss_func
        self.scales = scales
        self.kernel = kernel
        self.num_dim = num_dim
        if weights is None:
            weights = [1 / (2**i) for i in range(scales)]
        self.weights = weights

    def forward(self, moving, fixed):
        total = 0.0
        m, f = moving, fixed
        for i in range(self.scales):
            total = total + self.weights[i] * self.loss_func(m, f)
            if i != self.scales - 1:
                if self.num_dim == 2:
                    m = F.avg_pool2d(
                        m,
                        kernel_size=self.kernel,
                        stride=2,
                        padding=self.kernel // 2,
                        count_include_pad=False,
                    )
                    f = F.avg_pool2d(
                        f,
                        kernel_size=self.kernel,
                        stride=2,
                        padding=self.kernel // 2,
                        count_include_pad=False,
                    )
                elif self.num_dim == 3:
                    m = F.avg_pool3d(
                        m,
                        kernel_size=self.kernel,
                        stride=2,
                        padding=self.kernel // 2,
                        count_include_pad=False,
                    )
                    f = F.avg_pool3d(
                        f,
                        kernel_size=self.kernel,
                        stride=2,
                        padding=self.kernel // 2,
                        count_include_pad=False,
                    )
        return total


__all__ = ["LocalMutualInformation", "NGFLoss", "MultiScaleRegistrationLoss"]

