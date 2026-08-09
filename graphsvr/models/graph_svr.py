"""GraphSVR model: image encoders, acquisition graph, and rigid-motion GNN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm, JumpingKnowledge, TransformerConv

from graphsvr.geom.graph import sparse_a_matrix_stacks
from graphsvr.geom.rigid_transforms import transformationMatrices, world_mm_to_theta
from graphsvr.models.resnet3d import Encoder2D, generate_Resnet3D


class GraphSVR(nn.Module):
    """Predict one 6-DoF rigid transform per DWI slice group."""

    def __init__(
        self,
        rigid_config,
        GNN_args,
        features_size: int = 128,
        device: str = "cpu",
        k_for_a_matrix: int = 8,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if features_size <= 0:
            raise ValueError("features_size must be positive.")
        if k_for_a_matrix <= 0:
            raise ValueError("k_for_a_matrix must be positive.")

        self.dtype = dtype
        self.k_for_a_matrix = k_for_a_matrix
        self.rigid_parameters_ranges = torch.as_tensor(
            rigid_config["ranges"], dtype=dtype, device=device
        )
        if self.rigid_parameters_ranges.shape != (6, 2):
            raise ValueError(
                "rigid_config['ranges'] must have shape [6, 2], "
                f"got {tuple(self.rigid_parameters_ranges.shape)}"
            )
        self.add_to_scale = float(rigid_config["add_to_scale"])
        self.voxel_size = torch.as_tensor(rigid_config["voxel_size"], dtype=dtype)
        if self.voxel_size.numel() != 3:
            raise ValueError("rigid_config['voxel_size'] must contain three values.")

        self.StackEncoder = StackEncoder(n_features=features_size, d_in=1).to(dtype)
        self.VolumeEncoder = generate_Resnet3D(model_depth=4, n_classes=features_size).to(dtype)
        self.model = AttnNodeGNN(
            in_channels=features_size,
            edge_dim=1,
            **dict(GNN_args),
        ).to(dtype)

    def forward(
        self,
        dwi_stacks: torch.Tensor,
        q_space: torch.Tensor,
        stack_indices: torch.Tensor,
        t1_vol: torch.Tensor,
        timing: torch.Tensor,
        edges_components: dict,
        dwi_affine=None,
    ) -> torch.Tensor:
        """
        Args:
            dwi_stacks: ``[N, H, W, K]`` stack tensor.
            q_space: ``[N, 3]`` repeated diffusion directions.
            stack_indices: ``[N, K]`` z-slice indices for each stack.
            t1_vol: ``[H, W, D]`` anatomical reference in the DWI grid.
            timing: ``[N]`` acquisition-order values.
            edges_components: Flags selecting q-space/time/location graph terms.
            dwi_affine: 4x4 NIfTI affine.
        """
        if dwi_stacks.ndim != 4:
            raise ValueError(f"dwi_stacks must be [N,H,W,K], got {tuple(dwi_stacks.shape)}")
        n_stacks = dwi_stacks.shape[0]
        if q_space.shape != (n_stacks, 3):
            raise ValueError(f"q_space must be [N,3], got {tuple(q_space.shape)}")
        if stack_indices.ndim != 2 or stack_indices.shape[0] != n_stacks:
            raise ValueError(f"stack_indices must be [N,K], got {tuple(stack_indices.shape)}")
        if timing.shape != (n_stacks,):
            raise ValueError(f"timing must be [N], got {tuple(timing.shape)}")
        if t1_vol.ndim != 3:
            raise ValueError(f"t1_vol must be 3D, got {tuple(t1_vol.shape)}")
        if dwi_affine is None:
            dwi_affine = torch.eye(4, dtype=self.dtype, device=dwi_stacks.device)

        stack_features = self.StackEncoder(dwi_stacks).to(self.dtype)
        ref_features = self.VolumeEncoder(t1_vol.unsqueeze(0).unsqueeze(0)).to(self.dtype)
        super_idx = n_stacks
        node_features = torch.cat((stack_features, ref_features), dim=0)

        edge_index_ss, edge_weight_ss = sparse_a_matrix_stacks(
            q_space=q_space,
            slice_thickness=self.voxel_size[-1],
            slice_indices=stack_indices,
            timing=timing,
            k=self.k_for_a_matrix,
            **edges_components,
        )
        edge_weight_ss = edge_weight_ss.to(dtype=self.dtype, device=node_features.device)

        stack_nodes = torch.arange(n_stacks, device=node_features.device, dtype=torch.long)
        ref_nodes = torch.full_like(stack_nodes, super_idx)
        edge_to_ref = torch.stack((stack_nodes, ref_nodes), dim=0)
        edge_from_ref = torch.stack((ref_nodes, stack_nodes), dim=0)
        edge_index = torch.cat((edge_index_ss, edge_to_ref, edge_from_ref), dim=1)
        ref_weights = torch.ones(2 * n_stacks, device=node_features.device, dtype=self.dtype)
        edge_weight = torch.cat((edge_weight_ss, ref_weights), dim=0)

        rigid_params = self.model(node_features, edge_index, edge_weight.unsqueeze(1))[:n_stacks]
        pred_params = self.rescale_registration_net_output(rigid_params)
        rigid_trans_mm = transformationMatrices(
            pred_params[:, :3],
            pred_params[:, 3:],
            dwi_affine,
            t1_vol.shape,
            device=pred_params.device,
            dtype=self.dtype,
        )
        return world_mm_to_theta(rigid_trans_mm, dwi_affine, t1_vol.shape).to(self.dtype)

    def rescale_registration_net_output(self, rigid_params: torch.Tensor) -> torch.Tensor:
        """Map unconstrained GNN outputs into the configured rigid-parameter range."""
        ranges = self.rigid_parameters_ranges.to(
            device=rigid_params.device, dtype=rigid_params.dtype
        )
        scale = 1.0 + self.add_to_scale
        low = ranges[:, 0] * scale
        span = (ranges[:, 1] - ranges[:, 0]) * scale
        return low + torch.sigmoid(rigid_params.reshape(-1, 6)) * span


class AttnNodeGNN(nn.Module):
    """Attention-based message-passing network with jumping-knowledge aggregation."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden: int = 128,
        out_dim: int = 6,
        heads: int = 4,
        layers: int = 3,
        drop: float = 0.1,
    ):
        super().__init__()
        if hidden <= 0 or heads <= 0 or layers <= 0:
            raise ValueError("hidden, heads, and layers must be positive.")
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads.")
        if not 0.0 <= drop < 1.0:
            raise ValueError("drop must be in [0, 1).")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels,
                hidden // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=drop,
                beta=True,
            )
        )
        self.norms.append(GraphNorm(hidden))
        for _ in range(layers - 1):
            self.convs.append(
                TransformerConv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=drop,
                    beta=True,
                )
            )
            self.norms.append(GraphNorm(hidden))

        self.jk = JumpingKnowledge(mode="cat")
        self.head = nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        outputs = []
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(x, edge_index, edge_attr)))
            if h.shape == x.shape:
                h = h + x
            outputs.append(h)
            x = h
        return self.head(self.jk(outputs))


class StackEncoder(nn.Module):
    """Encode each slice independently and sum embeddings within a slice group."""

    def __init__(self, n_features: int = 128, d_in: int = 1):
        super().__init__()
        self.SliceEncoder = Encoder2D(d_model=n_features, d_in=d_in)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        n, h, w, k = stacks.shape
        slices = stacks.permute(0, 3, 1, 2).contiguous().reshape(n * k, 1, h, w)
        slice_features = self.SliceEncoder(slices).reshape(n, k, -1)
        return slice_features.sum(dim=1)


__all__ = ["GraphSVR"]
