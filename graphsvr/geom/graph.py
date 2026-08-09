"""Acquisition-graph construction used by GraphSVR."""

from __future__ import annotations

import torch
from torch_geometric.utils import add_self_loops, coalesce


def sparse_a_matrix_stacks(
    q_space: torch.Tensor,
    slice_thickness: torch.Tensor,
    slice_indices: torch.Tensor,
    timing: torch.Tensor,
    k: int = 4,
    eps: float = 1e-8,
    use_q_space: bool = True,
    use_timing: bool = True,
    use_x_space: bool = True,
):
    """Build the weighted, symmetrized kNN graph over q-space, z-location and time."""
    if q_space.ndim != 2 or q_space.shape[1] != 3:
        raise ValueError(f"q_space must have shape [N, 3], got {tuple(q_space.shape)}")
    n_nodes = q_space.shape[0]
    if n_nodes == 0:
        raise ValueError("Cannot build a graph with zero stack nodes.")
    if timing.shape != (n_nodes,):
        raise ValueError(f"timing must have shape [N], got {tuple(timing.shape)}")
    if slice_indices.ndim != 2 or slice_indices.shape[0] != n_nodes:
        raise ValueError(f"slice_indices must have shape [N, s], got {tuple(slice_indices.shape)}")
    if k <= 0:
        raise ValueError("k must be positive.")

    device, dtype = q_space.device, q_space.dtype
    timing = timing.to(device=device, dtype=dtype).reshape(n_nodes, 1)
    slice_indices = slice_indices.to(device=device, dtype=dtype)
    slice_thickness = torch.as_tensor(slice_thickness, device=device, dtype=dtype)
    z_mm = slice_indices * slice_thickness

    def zscore_cols(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - x.mean(dim=0, keepdim=True)) / std

    q_n = zscore_cols(q_space) if use_q_space else torch.zeros_like(q_space)
    z_n = zscore_cols(z_mm) if use_x_space else torch.zeros_like(z_mm)
    t_n = zscore_cols(timing) if use_timing else torch.zeros_like(timing)
    coords = torch.cat((q_n, z_n, t_n), dim=1)

    if n_nodes == 1:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_weight = torch.empty((0,), device=device, dtype=dtype)
    else:
        k_eff = min(k, n_nodes - 1)
        distances = torch.cdist(coords, coords, p=2)
        distances.fill_diagonal_(float("inf"))
        neighbors = distances.topk(k=k_eff, largest=False, dim=1).indices
        src = torch.arange(n_nodes, device=device).repeat_interleave(k_eff)
        dst = neighbors.reshape(-1)
        directed = torch.stack((src, dst), dim=0)
        edge_index = torch.cat((directed, directed.flip(0)), dim=1)
        edge_index = coalesce(edge_index, num_nodes=n_nodes)

        row, col = edge_index
        dist2 = (coords[row] - coords[col]).square().sum(dim=-1)
        positive = dist2[dist2 > eps]
        sigma2 = positive.median() if positive.numel() else torch.tensor(1.0, device=device, dtype=dtype)
        sigma2 = sigma2.clamp_min(1e-6)
        edge_weight = torch.exp(-dist2 / sigma2)

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=n_nodes
    )

    # Symmetric GCN normalization without requiring torch_scatter.
    degree = torch.zeros(n_nodes, device=device, dtype=dtype)
    degree.scatter_add_(0, edge_index[0], edge_weight)
    inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    norm_weight = inv_sqrt[edge_index[0]] * edge_weight * inv_sqrt[edge_index[1]]
    return edge_index, norm_weight


__all__ = ["sparse_a_matrix_stacks"]
