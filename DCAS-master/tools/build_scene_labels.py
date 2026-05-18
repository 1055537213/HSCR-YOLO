#!/usr/bin/env python
"""
Build a scene label mapping file for scene-guided DCAS training.

Examples
--------
1) Derive scene labels from parent folders:
   python tools/build_scene_labels.py --data ultralytics/datasets/VisDrone.yaml \
       --mode folder --parent-level 1

2) Normalize an existing csv/json/txt mapping:
   python tools/build_scene_labels.py --data ultralytics/datasets/VisDrone.yaml \
       --mode csv --source my_scene_labels.csv --image-key image --scene-key scene
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
try:
    import yaml
except ImportError as exc:  # pragma: no cover - script-level dependency check
    raise SystemExit("PyYAML is required to run this script. Install it with `pip install pyyaml`.") from exc
try:
    from scipy.cluster.vq import kmeans2, whiten
except ImportError as exc:  # pragma: no cover - script-level dependency check
    raise SystemExit("SciPy is required to run this script. Install it with `pip install scipy`.") from exc

IMG_EXTS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate scene_labels.txt for scene-guided DCAS training.')
    parser.add_argument('--data', required=True, help='Path to the detection dataset yaml.')
    parser.add_argument('--mode',
                        default='folder',
                        choices=('folder', 'csv', 'json', 'txt', 'pseudo'),
                        help='How to build scene labels.')
    parser.add_argument('--source',
                        help='Existing mapping file. Required for csv/json/txt modes. Ignored for folder mode.')
    parser.add_argument('--output',
                        help='Output scene label file. Defaults to <dataset_root>/scene_labels.txt.')
    parser.add_argument('--parent-level',
                        type=int,
                        default=1,
                        help='For folder mode, which ancestor folder to use as the scene name. 1 means the direct parent.')
    parser.add_argument('--image-key',
                        default='image',
                        help='Image column/key name when mode=csv or mode=json with object records.')
    parser.add_argument('--scene-key',
                        default='scene',
                        help='Scene column/key name when mode=csv or mode=json with object records.')
    parser.add_argument('--scene-names-out',
                        help='Optional file path to write a yaml snippet containing scene_names and scene_labels.')
    parser.add_argument('--splits',
                        nargs='*',
                        default=('train', 'val', 'test'),
                        help='Dataset splits to scan from the yaml.')
    parser.add_argument('--clusters',
                        type=int,
                        default=8,
                        help='Number of pseudo scene clusters when mode=pseudo.')
    parser.add_argument('--image-size',
                        type=int,
                        default=224,
                        help='Resize images to this size before extracting pseudo scene descriptors.')
    parser.add_argument('--sample-per-sequence',
                        type=int,
                        default=12,
                        help='Maximum frames to sample from each sequence when mode=pseudo.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used by pseudo clustering.')
    parser.add_argument('--min-cluster-images',
                        type=int,
                        default=200,
                        help='Merge pseudo clusters with fewer than this many images into the nearest large cluster.')
    return parser.parse_args()


def read_dataset_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['yaml_file'] = str(path.resolve())
    root = data.get('path') or path.parent
    root = Path(root)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data['path'] = root
    return data


def resolve_split_entries(root: Path, value) -> List[Path]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        entries = [value]
    resolved = []
    for entry in entries:
        p = Path(entry)
        if not p.is_absolute():
            p = (root / p).resolve()
        resolved.append(p)
    return resolved


def read_image_list_file(list_file: Path) -> List[Path]:
    items = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (list_file.parent / p).resolve()
            items.append(p)
    return items


def collect_images(entries: Sequence[Path]) -> List[Path]:
    images = []
    for entry in entries:
        if entry.is_dir():
            images.extend(sorted(p.resolve() for p in entry.rglob('*') if p.suffix.lower() in IMG_EXTS))
        elif entry.is_file():
            if entry.suffix.lower() in IMG_EXTS:
                images.append(entry.resolve())
            else:
                images.extend(read_image_list_file(entry))
    return sorted(dict.fromkeys(images))


def collect_dataset_images(data: dict, splits: Iterable[str]) -> List[Path]:
    images = []
    for split in splits:
        images.extend(collect_images(resolve_split_entries(data['path'], data.get(split))))
    return sorted(dict.fromkeys(images))


def rel_key(image_path: Path, dataset_root: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
        return str(rel).replace('\\', '/')
    except ValueError:
        return str(image_path.resolve()).replace('\\', '/')


def load_external_mapping(args, data: dict) -> Dict[str, str]:
    if not args.source:
        raise SystemExit(f'--source is required when mode={args.mode}')
    source = Path(args.source)
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.exists():
        raise SystemExit(f'Source mapping file not found: {source}')

    if args.mode == 'csv':
        with open(source, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {normalize_key(row[args.image_key], data['path']): str(row[args.scene_key]).strip() for row in rows}

    if args.mode == 'json':
        with open(source, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return {normalize_key(k, data['path']): str(v).strip() for k, v in payload.items()}
        if isinstance(payload, list):
            return {
                normalize_key(row[args.image_key], data['path']): str(row[args.scene_key]).strip()
                for row in payload
            }
        raise SystemExit('Unsupported json format. Expected {image: scene} or a list of records.')

    # txt mode
    mapping = {}
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                image, scene = line.rsplit(',', 1)
            else:
                image, scene = line.rsplit(None, 1)
            mapping[normalize_key(image, data['path'])] = scene.strip()
    return mapping


def normalize_key(value: str, dataset_root: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(dataset_root.resolve())
        except ValueError:
            p = p.resolve()
    return str(p).replace('\\', '/')


def derive_from_folders(images: Sequence[Path], dataset_root: Path, parent_level: int) -> Dict[str, str]:
    if parent_level < 1:
        raise SystemExit('--parent-level must be >= 1')
    mapping = {}
    for image_path in images:
        try:
            scene_name = image_path.parents[parent_level - 1].name
        except IndexError as exc:
            raise SystemExit(f'Cannot read parent level {parent_level} from {image_path}') from exc
        mapping[rel_key(image_path, dataset_root)] = scene_name
    return mapping


def match_external_mapping(images: Sequence[Path], dataset_root: Path, external: Dict[str, str]) -> Dict[str, str]:
    matched = {}
    for image_path in images:
        key = rel_key(image_path, dataset_root)
        candidates = {
            key,
            image_path.name,
            image_path.stem,
            str(image_path.resolve()).replace('\\', '/'),
        }
        value = None
        for candidate in candidates:
            if candidate in external:
                value = external[candidate]
                break
        if value is None:
            raise SystemExit(f'No scene label found for image: {image_path}')
        matched[key] = value
    return matched


def infer_sequence_id(image_path: Path) -> str:
    stem = image_path.stem
    return stem.split('_')[0] if '_' in stem else stem


def sample_sequence_frames(images: Sequence[Path], sample_per_sequence: int) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for image_path in images:
        grouped.setdefault(infer_sequence_id(image_path), []).append(image_path)
    sampled = {}
    for seq_id, seq_images in grouped.items():
        seq_images = sorted(seq_images)
        if sample_per_sequence <= 0 or len(seq_images) <= sample_per_sequence:
            sampled[seq_id] = seq_images
            continue
        indices = np.linspace(0, len(seq_images) - 1, sample_per_sequence, dtype=int)
        sampled[seq_id] = [seq_images[i] for i in indices]
    return sampled


def extract_scene_descriptor(image_path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f'Failed to read image: {image_path}')
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-8

    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    hsv_norm = hsv.astype(np.float32)
    hsv_norm[..., 0] /= 180.0
    hsv_norm[..., 1:] /= 255.0
    color_stats = np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1)), hsv_norm.mean((0, 1)), hsv_norm.std((0, 1))])

    edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
    edge_features = [edges.mean()]
    for grid_y in range(3):
        for grid_x in range(3):
            patch = edges[
                grid_y * image_size // 3:(grid_y + 1) * image_size // 3,
                grid_x * image_size // 3:(grid_x + 1) * image_size // 3
            ]
            edge_features.append(patch.mean())

    sat = hsv_norm[..., 1]
    val = hsv_norm[..., 2]
    percentile_features = np.array([
        np.percentile(sat, 10), np.percentile(sat, 50), np.percentile(sat, 90),
        np.percentile(val, 10), np.percentile(val, 50), np.percentile(val, 90),
    ], dtype=np.float32)

    return np.concatenate([hist.astype(np.float32), color_stats.astype(np.float32),
                           np.array(edge_features, dtype=np.float32), percentile_features], axis=0)


def _merge_small_clusters(labels: np.ndarray,
                          features_scaled: np.ndarray,
                          cluster_centers: np.ndarray,
                          seq_ids: Sequence[str],
                          sequence_samples: Dict[str, List[Path]],
                          min_cluster_images: int) -> np.ndarray:
    if min_cluster_images <= 0:
        return labels

    image_counts = {}
    for label, seq_id in zip(labels.tolist(), seq_ids):
        image_counts[label] = image_counts.get(label, 0) + len(sequence_samples[seq_id])

    valid_clusters = [label for label, count in image_counts.items() if count >= min_cluster_images]
    if not valid_clusters:
        return labels

    merged = labels.copy()
    for cluster_label, count in image_counts.items():
        if count >= min_cluster_images:
            continue
        member_idx = np.where(labels == cluster_label)[0]
        for idx in member_idx:
            distances = []
            for valid_label in valid_clusters:
                center = cluster_centers[valid_label]
                distances.append((np.linalg.norm(features_scaled[idx] - center), valid_label))
            merged[idx] = min(distances, key=lambda x: x[0])[1]
    return merged


def build_pseudo_mapping(images: Sequence[Path], dataset_root: Path, args) -> Dict[str, str]:
    np.random.seed(args.seed)
    sequence_samples = sample_sequence_frames(images, args.sample_per_sequence)

    seq_ids = []
    features = []
    for seq_id, seq_images in sequence_samples.items():
        descriptors = [extract_scene_descriptor(path, args.image_size) for path in seq_images]
        seq_ids.append(seq_id)
        features.append(np.mean(np.stack(descriptors, axis=0), axis=0))

    features = np.stack(features, axis=0).astype(np.float32)
    if len(features) == 0:
        raise SystemExit('No sequence features were extracted for pseudo scene labels.')

    num_clusters = max(2, min(args.clusters, len(features)))
    scaled = whiten(features)
    centroids, labels = kmeans2(scaled, num_clusters, minit='points', iter=50)
    labels = _merge_small_clusters(labels, scaled, centroids, seq_ids, sequence_samples, args.min_cluster_images)

    # Re-index remaining scene ids to consecutive scene_0..scene_n for cleaner yaml output.
    ordered_labels = sorted(dict.fromkeys(labels.tolist()))
    remap = {old: new for new, old in enumerate(ordered_labels)}
    seq_to_scene = {seq_id: f'scene_{remap[int(label)]}' for seq_id, label in zip(seq_ids, labels.tolist())}
    mapping = {}
    for image_path in images:
        mapping[rel_key(image_path, dataset_root)] = seq_to_scene[infer_sequence_id(image_path)]
    return mapping


def write_scene_labels(output: Path, mapping: Dict[str, str]):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for image_key in sorted(mapping):
            f.write(f'{image_key},{mapping[image_key]}\n')


def build_scene_names(mapping: Dict[str, str]) -> List[str]:
    return sorted(dict.fromkeys(mapping.values()))


def write_scene_yaml(output: Path, scene_names: List[str], label_path: Path, dataset_root: Path):
    rel_label = label_path.resolve()
    try:
        rel_label = rel_label.relative_to(dataset_root.resolve())
        rel_label_str = str(rel_label).replace('\\', '/')
    except ValueError:
        rel_label_str = str(rel_label).replace('\\', '/')
    snippet = {
        'scene_names': scene_names,
        'scene_labels': rel_label_str,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        yaml.safe_dump(snippet, f, sort_keys=False, allow_unicode=True)


def main():
    args = parse_args()
    data = read_dataset_yaml(Path(args.data).resolve())
    dataset_root = Path(data['path']).resolve()
    images = collect_dataset_images(data, args.splits)
    if not images:
        raise SystemExit('No images found from the provided dataset yaml.')

    if args.mode == 'folder':
        mapping = derive_from_folders(images, dataset_root, args.parent_level)
    elif args.mode == 'pseudo':
        mapping = build_pseudo_mapping(images, dataset_root, args)
    else:
        external = load_external_mapping(args, data)
        mapping = match_external_mapping(images, dataset_root, external)

    output = Path(args.output).resolve() if args.output else (dataset_root / 'scene_labels.txt')
    write_scene_labels(output, mapping)

    scene_names = build_scene_names(mapping)
    print(f'Wrote {len(mapping)} scene labels to {output}')
    print('scene_names:')
    for scene_name in scene_names:
        print(f'  - {scene_name}')
    print(f'scene_labels: {output}')

    if args.scene_names_out:
        write_scene_yaml(Path(args.scene_names_out).resolve(), scene_names, output, dataset_root)
        print(f'Wrote yaml snippet to {Path(args.scene_names_out).resolve()}')


if __name__ == '__main__':
    main()
