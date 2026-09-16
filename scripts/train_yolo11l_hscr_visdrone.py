from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DEFAULT_DATA = REPO_ROOT / "configs" / "VisDrone.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-hscr.yaml"
LOCAL_WEIGHT = REPO_ROOT / "weights" / "yolo11l.pt"


def default_pretrained() -> str:
    """Use a local YOLO11l weight if present, otherwise let Ultralytics download it."""
    return str(LOCAL_WEIGHT) if LOCAL_WEIGHT.exists() else "yolo11l.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HSCR-YOLO11l with Detect(P2,P3,P4) on VisDrone."
    )
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
    parser.add_argument("--name", default="yolo11l-hscr-visdrone-img640")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--fitness-metric", choices=("map50", "map50-95"), default="map50-95")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_directory = args.project / args.name
    if run_directory.exists():
        raise FileExistsError(f"Run directory already exists: {run_directory}. Choose a new --name.")
    model = YOLO(str(args.model))
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
        "cache": args.cache,
        "amp": args.amp,
        "exist_ok": False,
        "verbose": True,
    }
    print("Training HSCR-YOLO11l with Detect(P2,P3,P4) and retained P5 semantic context:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print(f"  model: {args.model}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
