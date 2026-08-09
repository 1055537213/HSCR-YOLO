import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.gsdr import GSDR, build_gsdr_targets
from ultralytics.utils import DEFAULT_CFG_DICT


class GSDRTargetTests(unittest.TestCase):
    def test_targets_have_expected_shape_and_density_order(self):
        batch_idx = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        boxes = torch.tensor(
            [
                [0.25, 0.50, 0.04, 0.04],
                [0.27, 0.50, 0.04, 0.04],
                [0.29, 0.50, 0.04, 0.04],
                [0.75, 0.75, 0.12, 0.10],
            ],
            dtype=torch.float32,
        )

        density, scale, scale_mask = build_gsdr_targets(
            batch_idx=batch_idx,
            boxes=boxes,
            batch_size=2,
            output_size=(16, 16),
        )

        self.assertEqual(density.shape, (2, 1, 16, 16))
        self.assertEqual(scale.shape, (2, 1, 16, 16))
        self.assertEqual(scale_mask.shape, (2, 1, 16, 16))
        self.assertTrue(torch.all((density >= 0) & (density <= 1)))
        self.assertTrue(torch.all((scale_mask >= 0) & (scale_mask <= 1)))

        crowded_value = density[0, 0, 8, 4]
        isolated_value = density[1, 0, 12, 12]
        self.assertGreater(crowded_value, isolated_value)
        self.assertGreater(scale_mask[0, 0, 8, 4], 0)
        self.assertGreater(scale_mask[1, 0, 12, 12], 0)


class GSDRModuleTests(unittest.TestCase):
    def test_routing_preserves_feature_shapes_and_gradients(self):
        module = GSDR([8, 16, 32], hidden_channels=8, warmup_epochs=3)
        module.train()
        module.set_epoch(0)
        self.assertAlmostEqual(module.warmup_factor(), 1 / 3)
        module.set_epoch(3)
        self.assertEqual(module.warmup_factor(), 1.0)
        features = [
            torch.randn(2, 8, 16, 16, requires_grad=True),
            torch.randn(2, 16, 8, 8, requires_grad=True),
            torch.randn(2, 32, 4, 4, requires_grad=True),
        ]

        routed = module(features)
        self.assertEqual([x.shape for x in routed], [x.shape for x in features])
        self.assertEqual(module.last_aux["density"].shape, (2, 1, 16, 16))
        self.assertEqual(module.last_aux["scale"].shape, (2, 1, 16, 16))

        sum(routed[0].mean() for _ in range(1)).backward()
        self.assertIsNotNone(features[0].grad)
        self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))


class GSDRIntegrationTests(unittest.TestCase):
    def test_gsdr_yaml_produces_detection_and_geometry_losses(self):
        yaml_path = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-gsdr.yaml"
        net = YOLO(str(yaml_path)).model
        net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        net.train()
        preds = net(torch.randn(2, 3, 128, 128))
        batch = {
            "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor(
                [[0.25, 0.50, 0.05, 0.05], [0.27, 0.50, 0.05, 0.05], [0.75, 0.75, 0.12, 0.10]]
            ),
        }

        loss, items = net.loss(batch, preds)
        self.assertTrue(net.has_gsdr)
        self.assertEqual(tuple(loss.shape), (5,))
        self.assertEqual(tuple(items.shape), (5,))
        self.assertTrue(torch.isfinite(loss).all())

        net.eval()
        with torch.no_grad():
            inference, raw = net(torch.randn(1, 3, 128, 128))
        self.assertEqual(inference.ndim, 3)
        self.assertIn("feats", raw)


if __name__ == "__main__":
    unittest.main()
