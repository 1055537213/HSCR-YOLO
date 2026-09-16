# YOLOv11-VisDrone Learning Resources

## Knowledge

- [PyTorch SGD documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
  Primary API reference for learning rate, momentum, weight decay, and parameter updates.
- [PyTorch Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
  Primary guide to `autocast`, gradient scaling, overflow checks, and optimizer stepping.
- [MobileNets paper](https://arxiv.org/abs/1704.04861)
  Original reference for depthwise separable convolutions and their efficiency motivation.
- [Ultralytics data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
  Official description of Mosaic probability and closing Mosaic near the end of training.
- [Local YOLO11 model configuration](ultralytics/cfg/models/11/yolo11.yaml)
  Repository source of the Backbone and P3/P4/P5 feature hierarchy.
- [Local GSDR implementation](ultralytics/nn/modules/gsdr.py)
  Repository source of the three dilated DWConv branches and routed residual update.

## Wisdom (Communities)

- [Ultralytics Discussions](https://github.com/orgs/ultralytics/discussions)
  Use for implementation-specific questions, reproducibility reports, and training tradeoffs.
