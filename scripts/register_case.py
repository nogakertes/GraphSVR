"""Command-line entry point for zero-shot GraphSVR registration."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# Make the repository importable even when this script is launched from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot GraphSVR registration for one DWI case."
    )

    parser.add_argument(
        "--case_path",
        type=Path,
        required=True,
        help="Case directory containing dwi.nii.gz, t1.nii.gz, dwi_mask.nii.gz, dwi.bval, dwi.bvec, and slspec.txt.",
    )
    parser.add_argument(
        "--save_exp_path",
        type=Path,
        required=True,
        help="Directory in which a timestamped result directory will be created.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="graphsvr_experiment",
        help="Experiment name used as the output-directory prefix.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or an explicit CUDA device such as 'cuda:0'.",
    )

    parser.add_argument(
        "--rigid_range_rxryrz",
        type=float,
        default=40.0,
        help="Symmetric rotation range in degrees for RX/RY/RZ.",
    )
    parser.add_argument(
        "--rigid_range_txtytz",
        type=float,
        default=40.0,
        help="Symmetric translation range in mm for TX/TY/TZ.",
    )
    parser.add_argument(
        "--add_to_scale",
        type=float,
        default=2.0,
        help="Additional scaling factor used by the rigid-parameter output mapping.",
    )

    parser.add_argument("--hidden", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--drop", type=float, default=0.01, help="Dropout probability.")
    parser.add_argument("--node_features_size", type=int, default=64, help="Node feature dimension.")

    parser.add_argument("--loss_weight_lmi", type=float, default=1.0, help="LMI loss weight.")
    parser.add_argument("--loss_weight_grad", type=float, default=1.0, help="NGF loss weight.")
    parser.add_argument(
        "--multi_scale_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the multi-scale registration objective.",
    )
    parser.add_argument(
        "--multi_scale_loss_scales",
        type=int,
        default=4,
        help="Number of multi-scale loss levels.",
    )
    parser.add_argument("--LMI_patch_size", type=int, default=5, help="LMI patch size.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=120, help="Number of stacks per optimization batch.")
    parser.add_argument("--num_epochs", type=int, default=400, help="Maximum optimization epochs.")
    parser.add_argument("--k_for_a_matrix", type=int, default=4, help="k for the stack kNN graph.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is the most portable default; increase on systems where multiprocessing is beneficial.",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )

    parsed = parser.parse_args(args)

    if parsed.hidden <= 0 or parsed.heads <= 0 or parsed.layers <= 0:
        parser.error("--hidden, --heads, and --layers must be positive")
    if parsed.hidden % parsed.heads != 0:
        parser.error("--hidden must be divisible by --heads")
    if not 0.0 <= parsed.drop < 1.0:
        parser.error("--drop must be in [0, 1)")
    if parsed.batch_size <= 0 or parsed.num_epochs <= 0 or parsed.k_for_a_matrix <= 0:
        parser.error("--batch_size, --num_epochs, and --k_for_a_matrix must be positive")
    if parsed.multi_scale_loss_scales <= 0 or parsed.LMI_patch_size <= 0:
        parser.error("loss scale count and LMI patch size must be positive")
    if parsed.num_workers < 0:
        parser.error("--num_workers must be >= 0")

    return parsed


def resolve_device(spec: str) -> torch.device:
    spec = spec.strip().lower()
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        spec = "cuda:0"

    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{spec}'.") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def main(args: Sequence[str] | None = None) -> None:
    cfg = parse_args(args)
    case_path = cfg.case_path.expanduser().resolve()
    save_path = cfg.save_exp_path.expanduser().resolve()

    if not case_path.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")
    set_seed(cfg.seed)

    rigid_stats = torch.tensor(
        [
            [-cfg.rigid_range_rxryrz, cfg.rigid_range_rxryrz],
            [-cfg.rigid_range_rxryrz, cfg.rigid_range_rxryrz],
            [-cfg.rigid_range_rxryrz, cfg.rigid_range_rxryrz],
            [-cfg.rigid_range_txtytz, cfg.rigid_range_txtytz],
            [-cfg.rigid_range_txtytz, cfg.rigid_range_txtytz],
            [-cfg.rigid_range_txtytz, cfg.rigid_range_txtytz],
        ],
        dtype=torch.float32,
        device=device,
    )
    rigid_config = {"ranges": rigid_stats, "add_to_scale": cfg.add_to_scale}
    gnn_args = {
        "hidden": cfg.hidden,
        "heads": cfg.heads,
        "layers": cfg.layers,
        "drop": cfg.drop,
    }
    loss_weights = {"LMI": cfg.loss_weight_lmi, "Grad": cfg.loss_weight_grad}

    # Import here so argument validation / --help remain available even if the
    # scientific Python environment has not yet been installed.
    from graphsvr.training.trainer import GraphSVRTrainer

    print(f"Starting experiment: {cfg.exp_name}")
    trainer = GraphSVRTrainer(
        str(case_path),
        rigid_config,
        gnn_args,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        node_features_size=cfg.node_features_size,
        loss_weights=loss_weights,
        save_exp_path=str(save_path),
        exp_name=cfg.exp_name,
        multi_scale_loss_scales=cfg.multi_scale_loss_scales,
        multi_scale_loss=cfg.multi_scale_loss,
        k_for_a_matrix=cfg.k_for_a_matrix,
        LMI_patch_size=cfg.LMI_patch_size,
        num_workers=cfg.num_workers,
        tensorboard=not cfg.no_tensorboard,
        device=str(device),
    )
    result_path = trainer.fit(num_epochs=cfg.num_epochs)
    print(f"Registration completed. Saved transforms to: {result_path}")


if __name__ == "__main__":
    main()
