"""Dataset loading and validation for GraphSVR registration cases."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from dipy.io import read_bvals_bvecs
from nibabel import load as load_nifti
from torch.utils.data import Dataset

REQUIRED_CASE_FILES = (
    "dwi.nii.gz",
    "t1.nii.gz",
    "dwi_mask.nii.gz",
    "dwi.bval",
    "dwi.bvec",
    "slspec.txt",
)


class GraphSVRDataset(Dataset):
    """Load one GraphSVR case and expose acquisition slice-groups as samples."""

    def __init__(self, case_path: str | Path, transform: Optional[Callable] = None):
        super().__init__()
        self.case_path = Path(case_path).expanduser().resolve()
        self.transform = transform
        self._validate_required_files()

        dwi_img = load_nifti(self.case_path / "dwi.nii.gz")
        t1_img = load_nifti(self.case_path / "t1.nii.gz")
        mask_img = load_nifti(self.case_path / "dwi_mask.nii.gz")

        if len(dwi_img.shape) != 4:
            raise ValueError(f"dwi.nii.gz must be 4D, got shape {dwi_img.shape}.")
        if len(t1_img.shape) != 3:
            raise ValueError(f"t1.nii.gz must be 3D, got shape {t1_img.shape}.")
        if len(mask_img.shape) != 3:
            raise ValueError(f"dwi_mask.nii.gz must be 3D, got shape {mask_img.shape}.")
        if tuple(dwi_img.shape[:3]) != tuple(t1_img.shape):
            raise ValueError(
                "T1 and DWI must have identical spatial dimensions. "
                f"Got DWI {dwi_img.shape[:3]} and T1 {t1_img.shape}."
            )
        if tuple(dwi_img.shape[:3]) != tuple(mask_img.shape):
            raise ValueError(
                "Mask and DWI must have identical spatial dimensions. "
                f"Got DWI {dwi_img.shape[:3]} and mask {mask_img.shape}."
            )
        if not np.allclose(dwi_img.affine, t1_img.affine, rtol=1e-4, atol=1e-3):
            raise ValueError("t1.nii.gz must be in the same coordinate system/affine as dwi.nii.gz.")
        if not np.allclose(dwi_img.affine, mask_img.affine, rtol=1e-4, atol=1e-3):
            raise ValueError("dwi_mask.nii.gz must be in the same coordinate system/affine as dwi.nii.gz.")

        stack_indices_np = np.loadtxt(self.case_path / "slspec.txt", dtype=np.int64)
        stack_indices_np = np.atleast_2d(stack_indices_np)
        if stack_indices_np.size == 0 or stack_indices_np.shape[1] == 0:
            raise ValueError("slspec.txt is empty or malformed.")
        if stack_indices_np.min() < 0 or stack_indices_np.max() >= dwi_img.shape[2]:
            raise ValueError(
                "slspec.txt contains slice indices outside the DWI z dimension "
                f"[0, {dwi_img.shape[2] - 1}]."
            )
        flattened = np.sort(stack_indices_np.reshape(-1))
        expected = np.arange(dwi_img.shape[2])
        if flattened.shape != expected.shape or not np.array_equal(flattened, expected):
            raise ValueError(
                "slspec.txt must list every DWI slice exactly once across its rows. "
                "Use the same slice-group specification format as FSL eddy."
            )
        self.stack_indices = torch.from_numpy(stack_indices_np)

        bvals_np, bvecs_np = read_bvals_bvecs(
            str(self.case_path / "dwi.bval"), str(self.case_path / "dwi.bvec")
        )
        if bvals_np is None or bvecs_np is None:
            raise ValueError("Could not parse dwi.bval/dwi.bvec.")
        if len(bvals_np) != dwi_img.shape[3] or bvecs_np.shape != (dwi_img.shape[3], 3):
            raise ValueError(
                "Gradient table length must match the number of DWI volumes. "
                f"Got {len(bvals_np)} b-values, b-vectors shape {bvecs_np.shape}, "
                f"and {dwi_img.shape[3]} DWI volumes."
            )
        if not np.isfinite(bvals_np).all() or not np.isfinite(bvecs_np).all():
            raise ValueError("dwi.bval/dwi.bvec contain non-finite values.")

        self.voxel_size = torch.as_tensor(dwi_img.header.get_zooms()[:3], dtype=torch.float32)
        temporal_zoom = dwi_img.header.get_zooms()[3]
        self.dt = torch.tensor(float(temporal_zoom), dtype=torch.float32)
        self.affine = torch.as_tensor(dwi_img.affine, dtype=torch.float32)

        self.t1 = torch.from_numpy(t1_img.get_fdata(dtype=np.float32)).contiguous()
        self.mask = torch.from_numpy(mask_img.get_fdata(dtype=np.float32)).contiguous()
        if not torch.isfinite(self.t1).all():
            raise ValueError("t1.nii.gz contains NaN or infinite values.")
        if not torch.isfinite(self.mask).all():
            raise ValueError("dwi_mask.nii.gz contains NaN or infinite values.")

        dwi_data = dwi_img.get_fdata(dtype=np.float32)
        if not np.isfinite(dwi_data).all():
            raise ValueError("dwi.nii.gz contains NaN or infinite values.")

        # [volume, stack, x, y, mb] -> [all_stacks, x, y, mb]
        stacks = [
            torch.from_numpy(dwi_data[..., group, volume])
            for volume in range(dwi_data.shape[-1])
            for group in stack_indices_np
        ]
        self.dwi_stacks = torch.stack(stacks, dim=0).contiguous()

        self.bvals = torch.as_tensor(bvals_np, dtype=self.dwi_stacks.dtype)
        self.bvecs = torch.as_tensor(bvecs_np, dtype=self.dwi_stacks.dtype)
        n_stacks_in_vol = stack_indices_np.shape[0]
        self.bvecs_repeated = self.bvecs.repeat_interleave(n_stacks_in_vol, dim=0).contiguous()
        self.bvals_repeated = self.bvals.repeat_interleave(n_stacks_in_vol, dim=0).contiguous()
        self.stack_indices_repeated = self.stack_indices.repeat(dwi_img.shape[3], 1).contiguous()
        self.timing = torch.arange(self.dwi_stacks.shape[0], dtype=torch.float32) * float(self.dt)

        self._t1_tf = None
        self._mask_tf = None

    def _validate_required_files(self) -> None:
        if not self.case_path.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {self.case_path}")
        missing = [name for name in REQUIRED_CASE_FILES if not (self.case_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Case directory is missing required file(s): {', '.join(missing)}"
            )

    def __len__(self) -> int:
        return self.dwi_stacks.shape[0]

    def _get_shared(self):
        if self.transform is None:
            return self.mask, self.t1
        if self._t1_tf is None:
            self._t1_tf = self.transform(self.t1).contiguous()
        if self._mask_tf is None:
            # Keep the mask binary/discrete instead of intensity-normalizing it.
            self._mask_tf = self.mask
        return self._mask_tf, self._t1_tf

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dwi_stack = self.dwi_stacks[index]
        if self.transform is not None:
            dwi_stack = self.transform(dwi_stack)
        mask, t1 = self._get_shared()

        return (
            dwi_stack,
            self.bvecs_repeated[index],
            self.bvals_repeated[index],
            mask,
            t1,
            self.stack_indices_repeated[index],
            self.timing[index],
        )


def collate_keep_single(batch):
    """Collate per-stack tensors while keeping shared T1/mask tensors single."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    cols = list(zip(*batch))
    if len(cols) != 7:
        raise ValueError(f"Expected 7 elements per sample, got {len(cols)}.")

    return (
        torch.stack(cols[0], dim=0),
        torch.stack(cols[1], dim=0),
        torch.stack(cols[2], dim=0),
        cols[3][0],
        cols[4][0],
        torch.stack(cols[5], dim=0),
        torch.stack(cols[6], dim=0),
    )


__all__ = ["GraphSVRDataset", "collate_keep_single", "REQUIRED_CASE_FILES"]
