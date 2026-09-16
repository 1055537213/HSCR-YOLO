import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from ultralytics.nn.modules.gsdr import GSDR
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from scripts import train_yolo11l_gsdr_visdrone as training
from scripts import train_yolo11l_hscr_visdrone as hscr_training
from scripts import train_yolo11l_visdrone as baseline_training


MODEL_ROOT = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/11"


class GSDRAblationTests(unittest.TestCase):
    def test_default_experiment_is_uniform_640_batch8_not_v6(self):
        with patch("sys.argv", ["train"]):
            args = training.parse_args()
        self.assertEqual(args.model, MODEL_ROOT / "yolo11l-gsdr-v4-uniform.yaml")
        self.assertEqual(args.name, "yolo11l-gsdr-v4-uniform-visdrone-img640-b8-s0")
        self.assertEqual(args.seed, 0)
        cfg = YAML.load(args.model)
        self.assertTrue(cfg["head"][-2][3][-1])

    def test_uniform_forward_preserves_gate_and_trains_all_context_branches(self):
        module = GSDR([8, 16, 32], hidden_channels=8, uniform_routing=True).eval()
        features = [
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 16, 4, 4),
            torch.randn(2, 32, 2, 2),
        ]
        p2 = features[0]
        prior = module.prior_head(module.prior_stem(p2))
        density = prior[:, :1].sigmoid()
        projected = module.p2_in_proj(p2)
        context = sum(branch(projected) for branch in module.p2_context_branches) / 3
        alpha = module.alpha_max * module.residual_logits[0].sigmoid()
        expected = p2 + alpha * density * module.p2_out_proj(context)
        routed = module(features)
        torch.testing.assert_close(routed[0], expected)
        self.assertIs(routed[1], features[1])
        self.assertIs(routed[2], features[2])
        routed[0].square().mean().backward()
        for branch in module.p2_context_branches:
            self.assertGreater(branch[0].conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.prior_head.weight.grad[0].abs().sum().item(), 0)
        self.assertIsNone(module.density_gap_logits.grad)

    def test_legacy_checkpoint_without_uniform_flag_preserves_dynamic_routing(self):
        module = GSDR([8, 16, 32], hidden_channels=8, use_scale_for_routing=True)
        density = torch.tensor([0.05, 0.175, 0.30]).view(1, 1, 1, 3)
        scale = torch.full_like(density, 0.3)
        expected = module.routing_weights(density, scale)
        del module.uniform_routing
        torch.testing.assert_close(module.routing_weights(density, scale), expected)

    def test_yaml_pair_matches_initial_weights_and_survives_reconstruction(self):
        dynamic_cfg = YAML.load(MODEL_ROOT / "yolo11l-gsdr-v4-dynamic.yaml")
        uniform_cfg = YAML.load(MODEL_ROOT / "yolo11l-gsdr-v4-uniform.yaml")
        expected = deepcopy(dynamic_cfg)
        expected["head"][-2][3][-1] = True
        self.assertEqual(uniform_cfg, expected)

        models = []
        for cfg in (dynamic_cfg, uniform_cfg):
            torch.manual_seed(0)
            models.append(DetectionModel(cfg, nc=10, verbose=False))
        dynamic, uniform = models
        self.assertEqual(dynamic.state_dict().keys(), uniform.state_dict().keys())
        for key, value in dynamic.state_dict().items():
            torch.testing.assert_close(value, uniform.state_dict()[key], rtol=0, atol=0)
        rebuilt = DetectionModel(deepcopy(uniform.yaml), nc=10, verbose=False)
        module = next(m for m in rebuilt.modules() if isinstance(m, GSDR))
        self.assertTrue(module.uniform_routing)
        self.assertFalse(module.use_scale_for_routing)
        self.assertFalse(module.detach_density_for_routing)
        self.assertFalse(module.calibrate_residual_gate)
        weights = module.routing_weights(torch.rand(2, 1, 8, 8))
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))

    def test_entrypoint_passes_seed_and_rejects_existing_result_directory(self):
        for entrypoint in (training, hscr_training, baseline_training):
            with self.subTest(entrypoint=entrypoint.__name__):
                self.check_entrypoint(entrypoint)

    def check_entrypoint(self, entrypoint):
        with TemporaryDirectory() as directory:
            with patch("sys.argv", ["train", "--seed", "2", "--project", directory, "--name", "paired"]):
                args = entrypoint.parse_args()
            with patch.object(entrypoint, "parse_args", return_value=args), patch.object(entrypoint, "YOLO") as yolo:
                entrypoint.main()
                kwargs = yolo.return_value.train.call_args.kwargs
                self.assertEqual(kwargs["imgsz"], 640)
                self.assertEqual(kwargs["batch"], 16 if entrypoint is baseline_training else 8)
                self.assertEqual(kwargs["fitness_metric"], "map50-95")
                self.assertEqual(kwargs["save_period"], -1)
                self.assertEqual(kwargs["seed"], 2)
                self.assertTrue(kwargs["deterministic"])
                self.assertFalse(kwargs["exist_ok"])
                (Path(directory) / "paired").mkdir()
                with self.assertRaises(FileExistsError):
                    entrypoint.main()
                self.assertEqual(yolo.call_count, 1)


if __name__ == "__main__":
    unittest.main()
