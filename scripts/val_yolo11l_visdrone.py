from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DEFAULT_DATA = REPO_ROOT / "configs" / "VisDrone.yaml"
DEFAULT_MODEL = REPO_ROOT / "runs" / "train" / "VisDrone" / "yolo11l-visdrone-img832" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO11l VisDrone checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Checkpoint path.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset yaml path.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "val" / "VisDrone")
    parser.add_argument("--name", default="yolo11l-visdrone-img832")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print(metrics)


if __name__ == "__main__":
    main()
