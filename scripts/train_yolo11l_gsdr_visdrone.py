from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.gsdr import GSDR


DEFAULT_DATA = REPO_ROOT / "configs" / "VisDrone.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-gsdr.yaml"
LOCAL_WEIGHT = REPO_ROOT / "weights" / "yolo11l.pt"


def default_pretrained() -> str:
    """Use a local YOLO11l weight if present, otherwise let Ultralytics download it."""
    return str(LOCAL_WEIGHT) if LOCAL_WEIGHT.exists() else "yolo11l.pt"


def set_gsdr_epoch(trainer) -> None:
    """Synchronize GSDR's short warmup with the trainer epoch."""
    model = getattr(trainer.model, "module", trainer.model)
    for module in model.modules():
        if isinstance(module, GSDR):
            module.set_epoch(trainer.epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GSDR-YOLO11l on VisDrone.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pretrained-weights", default=default_pretrained())
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "train" / "VisDrone")
    parser.add_argument("--name", default="yolo11l-gsdr-visdrone-img832")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--fitness-metric", choices=("map50", "map50-95"), default="map50")
    parser.add_argument("--gsdr-density-gain", type=float, default=0.2)
    parser.add_argument("--gsdr-scale-gain", type=float, default=0.1)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    model.add_callback("on_train_epoch_start", set_gsdr_epoch)
    train_args = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
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
        "cache": args.cache,
        "amp": args.amp,
        "exist_ok": True,
        "verbose": True,
    }
    print("Training GSDR-YOLO11l with geometry-supervised density-scale dual routing:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print(f"  model: {args.model}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
