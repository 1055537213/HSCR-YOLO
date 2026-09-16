from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.gsdr import GSDR
from ultralytics.utils import LOGGER, RANK


DEFAULT_DATA = REPO_ROOT / "configs" / "VisDrone.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-gsdr-v4-uniform.yaml"
LOCAL_WEIGHT = REPO_ROOT / "weights" / "yolo11l.pt"


def default_pretrained() -> str:
    """Use a local YOLO11l weight if present, otherwise let Ultralytics download it."""
    return str(LOCAL_WEIGHT) if LOCAL_WEIGHT.exists() else "yolo11l.pt"


def validate_training_device(device: str, world_size: int | None = None) -> None:
    """Reject multi-process training because GSDR epoch callbacks are process-local."""
    world_size = int(os.getenv("WORLD_SIZE", "1")) if world_size is None else int(world_size)
    requested_devices = [part.strip() for part in str(device).strip("[]").split(",") if part.strip()]
    if world_size > 1 or len(requested_devices) > 1:
        raise ValueError("GSDR training currently requires one process and one device so warmup and diagnostics stay valid.")


def set_gsdr_epoch(trainer) -> None:
    """Synchronize GSDR's residual warmup and reset epoch diagnostics."""
    model = getattr(trainer.model, "module", trainer.model)
    for module in model.modules():
        if isinstance(module, GSDR):
            module.set_epoch(trainer.epoch)
            module.reset_diagnostics()


def collect_gsdr_diagnostics(trainer) -> None:
    """Collect training-only routing statistics once per epoch."""
    model = getattr(trainer.model, "module", trainer.model)
    diagnostics = {}
    for module in model.modules():
        if isinstance(module, GSDR):
            diagnostics.update(module.consume_diagnostics())
    trainer.gsdr_epoch_diagnostics = {"epoch": trainer.epoch + 1, **diagnostics} if diagnostics else None
    if diagnostics:
        LOGGER.info(
            "GSDR routing: "
            f"soft=({diagnostics['route_soft_d1']:.3f}, {diagnostics['route_soft_d2']:.3f}, "
            f"{diagnostics['route_soft_d3']:.3f}), "
            f"positive=({diagnostics['positive_hard_d1']:.3f}, {diagnostics['positive_hard_d2']:.3f}, "
            f"{diagnostics['positive_hard_d3']:.3f}), "
            f"gate={diagnostics['residual_gate_mean']:.4f}, delta={diagnostics['p2_delta_rms_ratio']:.4f}"
        )


def save_gsdr_diagnostics(trainer) -> None:
    """Append the current epoch routing statistics to the run directory."""
    if RANK not in {-1, 0}:
        return
    row = getattr(trainer, "gsdr_epoch_diagnostics", None)
    if not row or row["epoch"] != trainer.epoch + 1:
        return
    if getattr(trainer, "_gsdr_last_logged_epoch", None) == row["epoch"]:
        return

    path = Path(trainer.save_dir) / "gsdr_diagnostics.csv"
    fresh_start = row["epoch"] == 1 and not bool(getattr(getattr(trainer, "args", None), "resume", False))
    write_header = fresh_start or not path.exists()
    with path.open("w" if fresh_start else "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    trainer._gsdr_last_logged_epoch = row["epoch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GSDR-YOLO11l on VisDrone.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pretrained-weights", default=default_pretrained())
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "train" / "VisDrone")
    parser.add_argument("--name", default="yolo11l-gsdr-v4-uniform-visdrone-img640-b8-s0")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--fitness-metric", choices=("map50", "map50-95"), default="map50-95")
    parser.add_argument("--gsdr-density-gain", type=float, default=0.1)
    parser.add_argument("--gsdr-scale-gain", type=float, default=0.05)
    parser.add_argument("--gsdr-density-positive-threshold", type=float, default=0.05)
    parser.add_argument(
        "--gsdr-density-hard-negative-ratio",
        type=float,
        default=3.0,
        help="Exponent controlling how strongly high-error background pixels are emphasized.",
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_training_device(args.device)
    run_directory = args.project / args.name
    if run_directory.exists():
        raise FileExistsError(f"Run directory already exists: {run_directory}. Choose a new --name.")
    model = YOLO(str(args.model))
    model.add_callback("on_train_epoch_start", set_gsdr_epoch)
    model.add_callback("on_train_epoch_end", collect_gsdr_diagnostics)
    model.add_callback("on_fit_epoch_end", save_gsdr_diagnostics)
    train_args = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "deterministic": True,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "optimizer": args.optimizer,
        "pretrained": args.pretrained_weights,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "save_period": args.save_period,
        "fitness_metric": args.fitness_metric,
        "gsdr_density_gain": args.gsdr_density_gain,
        "gsdr_scale_gain": args.gsdr_scale_gain,
        "gsdr_density_positive_threshold": args.gsdr_density_positive_threshold,
        "gsdr_density_hard_negative_ratio": args.gsdr_density_hard_negative_ratio,
        "cache": args.cache,
        "amp": args.amp,
        "exist_ok": False,
        "verbose": True,
    }
    print("Training VisDrone using the selected model configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print(f"  model: {args.model}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
