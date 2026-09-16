# GSDR mechanism experiment

This experiment tests whether v4's spatially adaptive branch selection improves detection.
It is a diagnostic experiment, not a new improved GSDR version or a promise of higher mAP.

## Matched configurations

| Group | Model YAML | Branch weights | Residual gate | Auxiliary losses |
| --- | --- | --- | --- | --- |
| R | Standard yolo11l.pt | No GSDR | No GSDR | None |
| A | yolo11l-hscr.yaml | No GSDR | No GSDR | None |
| B | yolo11l-gsdr-v4-uniform.yaml | 1/3, 1/3, 1/3 everywhere | Raw density | Density + scale |
| C | yolo11l-gsdr-v4-dynamic.yaml | Calibrated density routing (v4) | Raw density | Density + scale |

B and C have identical parameter tensors at matched initialization and execute all three
context branches. B's four routing-center parameters remain present but receive no gradient;
they do not influence its predictions. Its density head still receives detection gradients
through the residual gate. Scale prediction is auxiliary-only in both groups.

C versus B tests adaptive branch selection against equal mixing. B versus A tests the
combined extra convolution, gate and auxiliary supervision, not each contribution separately.
A failed C-versus-B result does not establish that every possible dynamic router is useless.
The v6 YAML and old checkpoint behavior remain available for historical reproduction,
but v6 is not scheduled for another run. A versus a matched R measures
the HSCR architecture change; GSDR versus A measures the added module's net benefit.
GSDR versus R alone cannot isolate GSDR's contribution.

## First screening run

All new training and validation use imgsz=640. Upload the changed Python files and model
YAML files. Keep the dataset split, labels, server environment and standard YOLO11l
pretrained weights identical. Do not initialize from a trained VisDrone checkpoint.

The historical `yolo11l-visdrone-img640` run used batch 16, seed 0, SGD, 300 maximum
epochs, patience 40, mosaic 0.3, close_mosaic 10, AMP, deterministic mode and workers 4.
Its saved best.pt records mAP50 0.38992, mAP50-95 0.22922 and fitness 0.22922.
Therefore the new main entry points select checkpoints and early-stop on mAP50-95,
not the mAP50 selection previously used for HSCR/GSDR at 832. Report both metrics
from that same selected checkpoint. Periodic checkpoint saving is disabled by default
(`save_period=-1`); best.pt and last.pt are still saved. This storage setting does not
change the training schedule or checkpoint selection metric.

Group A is already complete: `yolo11l-hscr-visdrone-img640`, batch 8, seed 0.
Its best.pt has precision 0.54548, recall 0.42838, mAP50 0.40748 and mAP50-95
0.23748 at epoch 115 (155 epochs completed). Both mAP maxima occur at that epoch.
Keep this run; do not retrain HSCR now. All new GSDR comparisons use batch 8.

The old R used batch 16, so it remains a historical reference, not a strict
architecture-only control for A. R does not need to be rerun before the GSDR screen.

Next run B, the uniform-routing control. This is now the training script's default:

```bash
cd /root/autodl-tmp/YOLOv11-VisDrone
python scripts/train_yolo11l_gsdr_visdrone.py --model ultralytics/cfg/models/11/yolo11l-gsdr-v4-uniform.yaml --name yolo11l-gsdr-v4-uniform-visdrone-img640-b8-s0 --seed 0 --batch 8 --imgsz 640 --fitness-metric map50-95 --device 0
```

After B, run C at the same settings to isolate adaptive branch selection.
Historical v4 at 832 cannot substitute for this dynamic control:

```bash
python scripts/train_yolo11l_gsdr_visdrone.py --model ultralytics/cfg/models/11/yolo11l-gsdr-v4-dynamic.yaml --name yolo11l-gsdr-v4-dynamic-visdrone-img640-b8-s0 --seed 0 --batch 8 --imgsz 640 --fitness-metric map50-95 --device 0
```

Existing output directories are rejected. Select a new name when changing model or seed.
Verify the printed model is the uniform YAML for B and the dynamic YAML for C;
both must use imgsz 640, batch 8, seed 0 and fitness_metric map50-95. Inspect saved
args.yaml too. Do not silently reduce batch for only one group. GSDR batch 8 has
not been memory-tested on the server; HSCR fitting does not guarantee GSDR fits.

Defer matched baseline replication until a candidate warrants confirmation. At that
stage use batch 8, a new name, and the same seed values as the compared groups:

```bash
python scripts/train_yolo11l_visdrone.py --name yolo11l-visdrone-img640-b8-s0 --seed 0 --batch 8 --imgsz 640 --fitness-metric map50-95
```

Do not use `--resume` to switch an old 832 experiment to 640. Run a fresh training job.
This protocol changes resolution and checkpoint selection together relative to the old
832 experiments; it cannot attribute a difference from those runs to resolution alone.

## Decision and confirmation

- If C is no better than B, adaptive routing has not earned further formula tuning.
- If B and C both fail to beat A, the current GSDR path has not demonstrated net benefit.
- If C beats B but barely beats A, the routing mechanism may help within GSDR while the
  module remains too weak to claim a meaningful detector improvement.
- If a candidate clearly beats A, rerun the selected comparisons with seeds 1 and 2.
  Treat three seeds as a first replication check, not conclusive statistical proof.

Use the same seed values in each compared group. A uses this same entry point with
`--model ultralytics/cfg/models/11/yolo11l-hscr.yaml`, an explicit run name and batch 8,
or the dedicated HSCR entry point with the same settings.
Record all outcomes, including failed seeds; do not select the best seed per group.

Before more training, use +0.005 absolute mAP50 (+0.5 percentage points) over the matched
HSCR mean as a provisional engineering target. This is a chosen usefulness threshold,
not statistical significance or a guaranteed improvement. Require non-decreasing mean
mAP50-95 at the selected best-mAP50-95 checkpoints and report paired deltas and variation.
Changing this target after seeing results must be declared as exploratory.

For the initial seed-0 screen, that target corresponds to mAP50 >= 0.41248 and
mAP50-95 >= 0.23748 relative to the existing A. Passing this screen is not proof
of a reproducible gain, and B beating A alone does not validate dynamic routing.

Also compare recall and precision at one fixed deployment confidence threshold, and
measure detection by actual bounding-box size. Class names are not object-size bins.
Best 10-epoch averages show curve stability but are not independent repeated trials.
Report parameter count and inference latency using the same hardware and input settings.
Repeated validation-set tuning needs a held-out labeled evaluation set for final claims.

## Diagnostics

`gsdr_diagnostics.csv` remains available. In group B, all soft weights must be 1/3.
Hard argmax reports dilation-1 for all pixels because of ties; that does NOT mean route
collapse, single-branch execution, or that the other two branches are inactive.
Use soft weights, P2 delta RMS and the actual detection metrics for this control.

Small feature-map probes describe internal behavior; they cannot prove the cause of an
mAP change or predict the gain from a new training run.
