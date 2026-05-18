from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from ultralytics import YOLO


DEFAULT_DATA = REPO_ROOT / 'ultralytics' / 'datasets' / 'VisDrone.yaml'
DEFAULT_MODEL = REPO_ROOT / 'ultralytics' / 'models' / 'v8' / 'yolov8+SCDDCAS.yaml'
GENERATED_DIR = REPO_ROOT / 'ultralytics' / 'datasets' / 'generated'
IMAGE_SUFFIXES = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got: {value}')


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_dataset_root(data_yaml: Path, config: dict) -> Path:
    root = Path(config.get('path') or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def resolve_split(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def normalize_key(path: str | Path) -> str:
    return str(Path(path)).replace('\\', '/')


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return normalize_key(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return normalize_key(path.resolve())


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p.resolve() for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    if path.is_file() and path.suffix.lower() == '.txt':
        items = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            item = Path(line)
            if not item.is_absolute():
                item = (path.parent / item).resolve()
            items.append(item)
        return items
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path.resolve()]
    raise FileNotFoundError(f'Unsupported split source: {path}')


def load_scene_mapping(scene_label_path: Path) -> dict[str, str]:
    mapping = {}
    for line in scene_label_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ',' in line:
            key, value = line.rsplit(',', 1)
        else:
            key, value = line.rsplit(None, 1)
        mapping[normalize_key(key)] = value.strip()
    return mapping


def write_list_file(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(items) + '\n', encoding='utf-8')


def build_smoke_dataset(
    data_yaml: Path,
    output_dir: Path,
    train_count: int,
    val_count: int,
) -> Path:
    config = load_yaml(data_yaml)
    root = resolve_dataset_root(data_yaml, config)
    train_images = collect_images(resolve_split(root, config.get('train')))[:train_count]
    val_images = collect_images(resolve_split(root, config.get('val')))[:val_count]

    if not train_images:
        raise RuntimeError(f'No training images found for smoke dataset under: {root}')
    if not val_images:
        raise RuntimeError(f'No validation images found for smoke dataset under: {root}')

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list = [normalize_key(path.resolve()) for path in train_images]
    val_list = [normalize_key(path.resolve()) for path in val_images]
    train_scene_keys = [rel_or_abs(path, root) for path in train_images]
    val_scene_keys = [rel_or_abs(path, root) for path in val_images]

    train_file = output_dir / 'train.txt'
    val_file = output_dir / 'val.txt'
    write_list_file(train_file, train_list)
    write_list_file(val_file, val_list)

    smoke_config = {
        'path': normalize_key(root.resolve()),
        'train': normalize_key(train_file.resolve()),
        'val': normalize_key(val_file.resolve()),
        'test': None,
        'names': config.get('names', {}),
    }

    scene_label_source = config.get('scene_labels')
    if config.get('scene_names'):
        smoke_config['scene_names'] = config['scene_names']

    if scene_label_source:
        scene_label_path = resolve_split(root, scene_label_source)
        scene_mapping = load_scene_mapping(scene_label_path)
        subset_labels = []
        for key in train_scene_keys + val_scene_keys:
            if key in scene_mapping:
                subset_labels.append(f'{key},{scene_mapping[key]}')
        if subset_labels:
            scene_label_file = output_dir / 'scene_labels.txt'
            write_list_file(scene_label_file, subset_labels)
            smoke_config['scene_labels'] = normalize_key(scene_label_file.resolve())

    smoke_yaml = output_dir / 'VisDrone-smoke.yaml'
    with open(smoke_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(smoke_config, f, sort_keys=False, allow_unicode=True)
    return smoke_yaml


def choose_device(device: str | None) -> str:
    if device:
        return str(device)
    return '0' if torch.cuda.is_available() else 'cpu'


def cuda_arch_supported(index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(index)
    return f'sm_{major}{minor}' in set(torch.cuda.get_arch_list())


def print_device_summary(device: str) -> None:
    if device == 'cpu':
        print('Using CPU for training.')
        return
    if not torch.cuda.is_available():
        print(f'CUDA is unavailable, falling back to device={device}.')
        return

    try:
        index = int(str(device).split(',')[0])
    except ValueError:
        index = 0

    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    supported_arches = set(torch.cuda.get_arch_list())
    sm_name = f'sm_{major}{minor}'
    print(f'Using CUDA device {index}: {name} ({sm_name})')
    if supported_arches and sm_name not in supported_arches:
        print('WARNING: current PyTorch build does not list this SM architecture. '
              'Smoke training may still run, but full training is safer after upgrading PyTorch/CUDA support.')


def build_model_reference(model_path: Path, scale: str) -> str:
    model_path = model_path.resolve()
    stem = model_path.stem
    if re.search(r'yolov\d+[nslmx]', stem):
        return str(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model yaml not found: {model_path}')
    scaled_name = re.sub(r'(yolov\d+)', rf'\1{scale}', model_path.name, count=1)
    return str(model_path.with_name(scaled_name))


def build_train_args(args, data_yaml: Path) -> dict:
    common = {
        'data': str(data_yaml.resolve()),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': str(args.project),
        'name': args.name,
        'exist_ok': True,
        'pretrained': False,
        'optimizer': args.optimizer,
        'close_mosaic': args.close_mosaic,
        'amp': args.amp,
        'scene': args.scene,
        'scene_supervision': args.scene_supervision,
        'mosaic': args.mosaic,
        'cache': args.cache,
        'verbose': True,
    }
    if args.profile == 'smoke':
        common.update({
            'save': False,
            'plots': False,
            'val': True,
        })
    return common


def parse_args():
    parser = argparse.ArgumentParser(description='Run Scene-guided DCAS training on VisDrone.')
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--device', default=None, help='Training device, e.g. 0 or cpu. Defaults to auto-select.')
    parser.add_argument('--project', type=Path, default=REPO_ROOT / 'runs' / 'train' / 'VisDrone')
    parser.add_argument('--name', default=None, help='Experiment name. Defaults to profile-based names.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--scale', choices=['n', 's', 'm', 'l', 'x'], default=None,
                        help='YOLO compound scale. Defaults to n for smoke and l for full.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--scene', type=float, default=0.0,
                        help='Scene classification loss weight. Set > 0 only when scene supervision is enabled.')
    parser.add_argument('--scene-supervision', type=parse_bool, default=False,
                        help='Enable auxiliary scene loss. Default keeps scene branch forward fusion only.')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--close-mosaic', type=int, default=None)
    parser.add_argument('--mosaic', type=float, default=None,
                        help='Mosaic probability. Lower values preserve scene supervision more often.')
    parser.add_argument('--cache', type=parse_bool, default=False)
    parser.add_argument('--amp', type=parse_bool, default=None)
    parser.add_argument('--smoke-train', type=int, default=32, help='Number of train images for smoke mode.')
    parser.add_argument('--smoke-val', type=int, default=16, help='Number of val images for smoke mode.')
    parser.add_argument('--smoke-dir', type=Path, default=GENERATED_DIR / 'visdrone_smoke')
    return parser.parse_args()


def apply_profile_defaults(args):
    if args.profile == 'smoke':
        args.epochs = args.epochs or 1
        args.batch = args.batch or 2
        args.scale = args.scale or 'n'
        args.workers = args.workers if args.workers is not None else 0
        args.close_mosaic = args.close_mosaic if args.close_mosaic is not None else 0
        args.mosaic = args.mosaic if args.mosaic is not None else 0.0
        args.amp = args.amp if args.amp is not None else False
        args.name = args.name or 'yolov8n+SCDDCAS-smoke'
    else:
        args.epochs = args.epochs or 300
        args.batch = args.batch or 16
        args.scale = args.scale or 'l'
        args.workers = args.workers if args.workers is not None else 4
        args.close_mosaic = args.close_mosaic if args.close_mosaic is not None else 10
        args.mosaic = args.mosaic if args.mosaic is not None else 0.3
        args.amp = args.amp if args.amp is not None else True
        args.name = args.name or 'yolov8l+SCDDCAS'
    if not args.scene_supervision:
        args.scene = 0.0
    device_explicit = args.device is not None
    args.device = choose_device(args.device)
    if args.profile == 'smoke' and not device_explicit and args.device != 'cpu' and not cuda_arch_supported():
        print('CUDA architecture is unsupported by the current PyTorch build; switching smoke run to CPU.')
        args.device = 'cpu'


def main():
    args = parse_args()
    apply_profile_defaults(args)

    model_ref = build_model_reference(args.model, args.scale)
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'Data yaml not found: {data_yaml}')

    if args.profile == 'smoke':
        data_yaml = build_smoke_dataset(data_yaml, args.smoke_dir.resolve(), args.smoke_train, args.smoke_val)
        print(f'Smoke dataset written to: {data_yaml}')

    print_device_summary(args.device)
    train_args = build_train_args(args, data_yaml)
    print('Training arguments:')
    for key, value in train_args.items():
        print(f'  {key}: {value}')
    print(f'  model: {model_ref}')

    model = YOLO(model_ref)
    model.train(**train_args)


if __name__ == '__main__':
    main()
