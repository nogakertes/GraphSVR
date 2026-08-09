"""Rigid transformation and volume-warping utilities used by GraphSVR."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def transformationMatrices(
    rotation_params_deg: torch.Tensor,
    translation_params_mm: torch.Tensor,
    nifti_affine,
    vol_size_WHD,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Create batched rigid world/mm transforms rotating around the image center."""
    if device is None:
        device = rotation_params_deg.device
    rot = rotation_params_deg.to(device=device, dtype=dtype)
    trans = translation_params_mm.to(device=device, dtype=dtype)
    if rot.ndim != 2 or rot.shape[1] != 3:
        raise ValueError(f"rotation_params_deg must be [N, 3], got {tuple(rot.shape)}")
    if trans.shape != rot.shape:
        raise ValueError(f"translation_params_mm must match rotation shape, got {tuple(trans.shape)}")
    if len(vol_size_WHD) != 3 or any(int(v) <= 1 for v in vol_size_WHD):
        raise ValueError(f"vol_size_WHD must contain three dimensions > 1, got {vol_size_WHD}")

    rot = torch.deg2rad(rot)
    rx, ry, rz = rot.unbind(dim=1)
    cosrx, cosry, cosrz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sinrx, sinry, sinrz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

    n = rot.shape[0]
    R = torch.zeros((n, 3, 3), device=device, dtype=dtype)
    R[:, 0, 0] = cosry * cosrz
    R[:, 0, 1] = sinrx * sinry * cosrz - cosrx * sinrz
    R[:, 0, 2] = cosrx * sinry * cosrz + sinrx * sinrz
    R[:, 1, 0] = cosry * sinrz
    R[:, 1, 1] = sinrx * sinry * sinrz + cosrx * cosrz
    R[:, 1, 2] = cosrx * sinry * sinrz - sinrx * cosrz
    R[:, 2, 0] = -sinry
    R[:, 2, 1] = sinrx * cosry
    R[:, 2, 2] = cosrx * cosry

    affine = torch.as_tensor(nifti_affine, device=device, dtype=dtype)
    if affine.shape != (4, 4):
        raise ValueError(f"nifti_affine must be [4, 4], got {tuple(affine.shape)}")
    W, H, D = (int(v) for v in vol_size_WHD)
    center_vox = torch.tensor(
        [(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0, 1.0],
        device=device,
        dtype=dtype,
    )
    center_world = (affine @ center_vox)[:3]

    transform = torch.eye(4, device=device, dtype=dtype).repeat(n, 1, 1)
    transform[:, :3, :3] = R
    transform[:, :3, 3] = trans
    to_center = torch.eye(4, device=device, dtype=dtype).repeat(n, 1, 1)
    from_center = torch.eye(4, device=device, dtype=dtype).repeat(n, 1, 1)
    to_center[:, :3, 3] = center_world
    from_center[:, :3, 3] = -center_world
    return to_center @ transform @ from_center


def world_mm_to_theta(
    T_world: torch.Tensor,
    nifti_affine,
    vol_size_WHD,
    align_corners: bool = True,
) -> torch.Tensor:
    """Convert batched world/mm transforms to PyTorch normalized grid transforms."""
    if T_world.ndim != 3 or T_world.shape[1:] != (4, 4):
        raise ValueError(f"T_world must have shape [N, 4, 4], got {tuple(T_world.shape)}")
    device, dtype = T_world.device, T_world.dtype
    n = T_world.shape[0]
    affine = torch.as_tensor(nifti_affine, device=device, dtype=dtype)
    if affine.shape != (4, 4):
        raise ValueError(f"nifti_affine must be [4, 4], got {tuple(affine.shape)}")
    affine = affine.unsqueeze(0).expand(n, -1, -1)
    T_vox = torch.linalg.inv(affine) @ T_world @ affine

    W, H, D = (int(v) for v in vol_size_WHD)
    if min(W, H, D) <= 1:
        raise ValueError("All spatial dimensions must be > 1.")
    if align_corners:
        scales = (2.0 / (W - 1), 2.0 / (H - 1), 2.0 / (D - 1))
        shifts = (-1.0, -1.0, -1.0)
    else:
        scales = (2.0 / W, 2.0 / H, 2.0 / D)
        shifts = (-1.0 + 1.0 / W, -1.0 + 1.0 / H, -1.0 + 1.0 / D)

    vox_to_norm = torch.eye(4, device=device, dtype=dtype).repeat(n, 1, 1)
    norm_to_vox = torch.eye(4, device=device, dtype=dtype).repeat(n, 1, 1)
    for axis, (scale, shift) in enumerate(zip(scales, shifts)):
        vox_to_norm[:, axis, axis] = scale
        vox_to_norm[:, axis, 3] = shift
        norm_to_vox[:, axis, axis] = 1.0 / scale
        norm_to_vox[:, axis, 3] = -shift / scale

    return vox_to_norm @ T_vox @ norm_to_vox


def wrap_3d_image_torch_batch(
    rigid_trans: torch.Tensor, images_batch: torch.Tensor
) -> torch.Tensor:
    """Warp a batch of [W,H,D] volumes using normalized 4x4 theta matrices."""
    if images_batch.ndim != 4:
        raise ValueError(f"images_batch must be [N, W, H, D], got {tuple(images_batch.shape)}")
    if rigid_trans.shape != (images_batch.shape[0], 4, 4):
        raise ValueError(
            f"rigid_trans must be [N, 4, 4] with matching N, got {tuple(rigid_trans.shape)}"
        )
    vol_dhw = images_batch.permute(0, 3, 2, 1).unsqueeze(1)
    grid = F.affine_grid(rigid_trans[:, :3, :], size=vol_dhw.shape, align_corners=True)
    out = F.grid_sample(
        vol_dhw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.squeeze(1).permute(0, 3, 2, 1)


__all__ = ["transformationMatrices", "world_mm_to_theta", "wrap_3d_image_torch_batch"]
