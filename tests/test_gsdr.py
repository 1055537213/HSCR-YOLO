import csv
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.gsdr import GSDR, build_gsdr_targets
from ultralytics.utils import DEFAULT_CFG_DICT

from scripts.train_yolo11l_gsdr_visdrone import (
    collect_gsdr_diagnostics,
    save_gsdr_diagnostics,
    validate_training_device,
)


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
    def test_density_calibration_spans_all_routing_branches(self):
        module = GSDR(
            [8, 16, 32],
            hidden_channels=8,
            density_route_floor=0.05,
            density_route_ceiling=0.30,
        )
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)

        calibrated = module.calibrate_density(density)
        route_weights = module.routing_weights(density)

        torch.testing.assert_close(calibrated.flatten(), torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(route_weights.argmax(dim=1).flatten().tolist(), [2, 1, 0])

    def test_v4_calibrates_branch_routing_but_uses_raw_density_for_residual_gate(self):
        module = GSDR(
            [8, 16, 32],
            hidden_channels=8,
            density_route_floor=0.05,
            density_route_ceiling=0.30,
            calibrate_residual_gate=False,
        ).eval()
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)

        torch.testing.assert_close(module.residual_gate(density), density)
        self.assertEqual(module.routing_weights(density).argmax(dim=1).flatten().tolist(), [2, 1, 0])

        features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 32, 4, 4),
        ]
        with patch.object(module, "residual_gate", side_effect=lambda value: torch.zeros_like(value)) as gate:
            routed = module(features)

        gate.assert_called_once()
        torch.testing.assert_close(routed[0], features[0])

    def test_v5_detaches_density_only_for_branch_routing(self):
        module = GSDR(
            [8, 16, 32],
            hidden_channels=8,
            calibrate_residual_gate=False,
            detach_density_for_routing=True,
        ).eval()
        density = torch.tensor([[[[0.175]]]], requires_grad=True)

        routing_input = module.routing_input(density)
        self.assertFalse(routing_input.requires_grad)
        self.assertIs(module.residual_gate(density), density)

        module.routing_weights(routing_input)[:, 0].sum().backward()
        self.assertIsNone(density.grad)
        self.assertIsNotNone(module.density_gap_logits.grad)
        self.assertGreater(float(module.density_gap_logits.grad.abs().sum()), 0.0)

        features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 32, 4, 4),
        ]
        with patch.object(module, "routing_input", wraps=module.routing_input) as routing_input_method:
            module(features)

        routing_input_method.assert_called_once()

    def test_v6_fuses_attached_density_and_inverse_scale_for_routing(self):
        module = GSDR(
            [8, 16, 32],
            hidden_channels=8,
            density_route_floor=0.05,
            density_route_ceiling=0.30,
            use_scale_for_routing=True,
        )
        density = torch.tensor([[[[0.05, 0.175, 0.175]]]], requires_grad=True)
        scale = torch.tensor([[[[0.20, 0.20, 0.80]]]], requires_grad=True)

        routing_signal = module.routing_signal(density, scale)
        route_weights = module.routing_weights(density, scale)

        torch.testing.assert_close(routing_signal.flatten(), torch.tensor([0.0, 0.6, 0.4]))
        self.assertEqual(route_weights.argmax(dim=1).flatten().tolist(), [2, 0, 1])
        route_weights[:, 0].sum().backward()
        self.assertGreater(float(density.grad.abs().sum()), 0.0)
        self.assertGreater(float(scale.grad.abs().sum()), 0.0)

    def test_scale_aware_routing_requires_matching_scale_map(self):
        module = GSDR([8, 16, 32], hidden_channels=8, use_scale_for_routing=True)
        density = torch.full((1, 1, 2, 2), 0.175)

        with self.assertRaises(ValueError):
            module.routing_weights(density)
        with self.assertRaises(ValueError):
            module.routing_weights(density, torch.full((1, 1, 1, 1), 0.5))

    def test_legacy_v4_without_stop_gradient_mode_keeps_attached_routing_density(self):
        module = GSDR([8, 16, 32], hidden_channels=8)
        del module.detach_density_for_routing
        density = torch.tensor([[[[0.175]]]], requires_grad=True)

        self.assertIs(module.routing_input(density), density)

    def test_legacy_checkpoint_without_scale_routing_mode_keeps_density_only_route(self):
        module = GSDR([8, 16, 32], hidden_channels=8)
        del module.use_scale_for_routing
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)
        scale = torch.tensor([0.90, 0.10, 0.90]).view(1, 1, 1, 3)

        torch.testing.assert_close(module.routing_signal(density, scale), module.calibrate_density(density))

    def test_legacy_v3_without_residual_gate_mode_keeps_calibrated_gate(self):
        module = GSDR([8, 16, 32], hidden_channels=8)
        del module.calibrate_residual_gate
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)

        torch.testing.assert_close(module.residual_gate(density).flatten(), torch.tensor([0.0, 0.5, 1.0]))

    def test_legacy_checkpoint_without_calibration_bounds_keeps_identity_routing(self):
        module = GSDR([8, 16, 32], hidden_channels=8)
        del module.density_route_floor
        del module.density_route_ceiling
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)

        torch.testing.assert_close(module.calibrate_density(density), density)

    def test_routing_diagnostics_accumulate_and_reset(self):
        module = GSDR([8, 16, 32], hidden_channels=8, warmup_epochs=0)
        module.train()
        features = [
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32, 4, 4),
        ]

        module(features)
        target = torch.zeros_like(module.last_aux["density"])
        target[:, :, :, :8] = 0.1
        module.record_target_diagnostics(module.last_aux["route_weights"], target, positive_threshold=0.05)
        diagnostics = module.consume_diagnostics()

        self.assertAlmostEqual(sum(diagnostics[f"route_soft_d{i}"] for i in range(1, 4)), 1.0, places=5)
        self.assertAlmostEqual(sum(diagnostics[f"positive_hard_d{i}"] for i in range(1, 4)), 1.0, places=5)
        self.assertAlmostEqual(sum(diagnostics[f"negative_hard_d{i}"] for i in range(1, 4)), 1.0, places=5)
        self.assertGreater(diagnostics["p2_delta_rms_ratio"], 0.0)
        self.assertGreaterEqual(diagnostics["residual_gate_mean"], 0.0)
        self.assertLessEqual(diagnostics["residual_gate_mean"], 1.0)
        self.assertEqual(module.consume_diagnostics(), {})

    def test_eval_forward_is_ema_deepcopy_safe(self):
        """ModelEMA copies the model after the construction-time eval forward pass."""
        module = GSDR([8, 16, 32], hidden_channels=8).eval()
        features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 32, 4, 4),
        ]

        module(features)
        deepcopy(module)
        self.assertIsNone(module.last_aux)

    def test_routing_preserves_feature_shapes_and_gradients(self):
        module = GSDR([8, 16, 32], hidden_channels=8, warmup_epochs=3)
        module.train()
        module.set_epoch(0)
        self.assertAlmostEqual(module.warmup_factor(), 1 / 3)
        module.set_epoch(3)
        self.assertEqual(module.warmup_factor(), 1.0)
        self.assertGreater(
            float(module.alpha_max * torch.sigmoid(module.residual_logits[0]) * module.warmup_factor()),
            0.01,
        )
        features = [
            torch.randn(2, 8, 16, 16, requires_grad=True),
            torch.randn(2, 16, 8, 8, requires_grad=True),
            torch.randn(2, 32, 4, 4, requires_grad=True),
        ]

        routed = module(features)
        self.assertEqual([x.shape for x in routed], [x.shape for x in features])
        self.assertIs(routed[1], features[1])
        self.assertIs(routed[2], features[2])
        self.assertEqual(module.last_aux["density"].shape, (2, 1, 16, 16))
        self.assertEqual(module.last_aux["scale"].shape, (2, 1, 16, 16))

        sum(routed[0].mean() for _ in range(1)).backward()
        self.assertIsNotNone(features[0].grad)
        self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))
        self.assertIsNotNone(module.consume_aux())
        self.assertIsNone(module.last_aux)


class GSDRConfigTests(unittest.TestCase):
    def test_training_script_rejects_multi_process_devices(self):
        validate_training_device("0", world_size=1)
        with self.assertRaises(ValueError):
            validate_training_device("0,1", world_size=1)
        with self.assertRaises(ValueError):
            validate_training_device("0", world_size=2)

    def test_training_overrides_are_registered(self):
        config = get_cfg(
            overrides={
                "gsdr_density_gain": 0.1,
                "gsdr_scale_gain": 0.05,
                "gsdr_density_positive_threshold": 0.05,
                "gsdr_density_hard_negative_ratio": 3.0,
            }
        )
        self.assertEqual(config.gsdr_density_positive_threshold, 0.05)
        self.assertEqual(config.gsdr_density_hard_negative_ratio, 3.0)
        with self.assertRaises(ValueError):
            get_cfg(overrides={"gsdr_density_positive_threshold": 1.1})


class GSDRDiagnosticsCallbackTests(unittest.TestCase):
    def test_epoch_diagnostics_are_written_once(self):
        module = GSDR([8, 16, 32], hidden_channels=8, warmup_epochs=0).train()
        features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 32, 4, 4),
        ]
        module(features)
        target = torch.zeros_like(module.last_aux["density"])
        target[:, :, :8] = 0.1
        module.record_target_diagnostics(module.last_aux["route_weights"], target, positive_threshold=0.05)

        with TemporaryDirectory() as directory:
            diagnostics_path = Path(directory) / "gsdr_diagnostics.csv"
            diagnostics_path.write_text("epoch,stale\n99,1\n", encoding="utf-8")
            trainer = SimpleNamespace(
                model=module,
                epoch=0,
                save_dir=Path(directory),
                args=SimpleNamespace(resume=False),
            )
            collect_gsdr_diagnostics(trainer)
            save_gsdr_diagnostics(trainer)
            save_gsdr_diagnostics(trainer)

            with diagnostics_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["epoch"]), 1)
        self.assertAlmostEqual(sum(float(rows[0][f"route_soft_d{i}"]) for i in range(1, 4)), 1.0, places=5)


class GSDRIntegrationTests(unittest.TestCase):
    def test_gsdr_yaml_produces_detection_and_geometry_losses(self):
        yaml_path = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "11" / "yolo11l-gsdr.yaml"
        net = YOLO(str(yaml_path)).model
        net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        net.train()
        preds = net(torch.randn(2, 3, 128, 128))
        gsdr = next(module for module in net.modules() if isinstance(module, GSDR))
        self.assertTrue(gsdr.last_aux["density"].requires_grad)
        self.assertTrue(gsdr.last_aux["scale"].requires_grad)
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
        self.assertFalse(gsdr.calibrate_residual_gate)
        self.assertFalse(gsdr.detach_density_for_routing)
        self.assertTrue(gsdr.use_scale_for_routing)
        self.assertIsNone(gsdr.last_aux)
        diagnostics = gsdr.consume_diagnostics()
        self.assertGreater(diagnostics["positive_fraction"], 0.0)
        self.assertAlmostEqual(sum(diagnostics[f"positive_hard_d{i}"] for i in range(1, 4)), 1.0, places=5)
        self.assertAlmostEqual(sum(diagnostics[f"negative_hard_d{i}"] for i in range(1, 4)), 1.0, places=5)
        loss.sum().backward()
        self.assertIsNotNone(gsdr.p2_out_proj.conv.weight.grad)
        self.assertIsNotNone(gsdr.prior_head.weight.grad)
        self.assertIsNotNone(gsdr.residual_logits.grad)
        self.assertTrue(torch.isfinite(gsdr.residual_logits.grad).all())

        empty_target_prediction = torch.full((2, 1, 8, 8), 0.1, requires_grad=True)
        empty_target = torch.zeros_like(empty_target_prediction)
        background_loss = net.criterion._balanced_density_loss(empty_target_prediction, empty_target)
        self.assertTrue(torch.isfinite(background_loss))
        background_loss.backward()
        self.assertIsNotNone(empty_target_prediction.grad)

        net.eval()
        with torch.no_grad():
            inference, raw = net(torch.randn(1, 3, 128, 128))
        self.assertEqual(inference.ndim, 3)
        self.assertIn("feats", raw)


if __name__ == "__main__":
    unittest.main()
