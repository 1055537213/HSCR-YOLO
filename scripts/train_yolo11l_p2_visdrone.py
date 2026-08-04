from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DEFAULT_DATA = REPO_ROOT / "configs" / "VisDrone.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-p2.yaml"
LOCAL_WEIGHT = REPO_ROOT / "weights" / "yolo11l.pt"


def default_pretrained() -> str:
    """Use a local YOLO11l weight if present, otherwise let Ultralytics download it."""
    return str(LOCAL_WEIGHT) if LOCAL_WEIGHT.exists() else "yolo11l.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11l-P2 on VisDrone.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO11-P2 model yaml.")
    parser.add_argument("--pretrained-weights", default=default_pretrained(), help="YOLO11l weights for transfer.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "train" / "VisDrone")
    parser.add_argument("--name", default="yolo11l-p2-visdrone-img832")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
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
        "cache": args.cache,
        "amp": args.amp,
        "exist_ok": True,
        "verbose": True,
    }
    print("Training YOLO11l-P2 with arguments:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print(f"  model: {args.model}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
