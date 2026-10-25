"""Run with: uv run --no-project --with pyyaml python modular-max/test_spec.py"""

from pathlib import Path
import unittest

import yaml


class MAXSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = yaml.safe_load(Path(__file__).with_name("spec.yaml").read_text())

    def test_default_and_api_types(self):
        self.assertEqual(self.spec["default_version"], "26.7.0-cuda")
        self.assertEqual(self.spec["health_check_path"], "/v1/health")
        self.assertEqual(
            yaml.safe_load(self.spec["default_env"]["MAX_SERVE_API_TYPES"]),
            ["openai", "responses"],
        )

    def test_versions_images_commands_and_frameworks(self):
        images = {
            "cuda": ("max-nvidia-full", "gpu:all"),
            "rocm": ("max-amd", "gpu:all"),
            "cpu": ("max-nvidia-base", "cpu"),
        }
        expected_versions = {
            f"{release}-{framework}"
            for release in ("26.7.0", "26.6.0")
            for framework in images
        }
        self.assertEqual(set(self.spec["version_configs"]), expected_versions)
        for framework, (image, device) in images.items():
            expected = [f"{release}-{framework}" for release in ("26.7.0", "26.6.0")]
            self.assertEqual(self.spec["framework_index_map"][framework], expected)
            for version in expected:
                with self.subTest(version=version):
                    config = self.spec["version_configs"][version]
                    release = version.rsplit("-", 1)[0]
                    self.assertEqual(config["image_name"], f"modular/{image}:{release}")
                    self.assertEqual(config["custom_framework"], framework)
                    self.assertEqual(
                        config["run_command"],
                        "--model {{model_path}} --port {{port}} "
                        "--served-model-name {{model_name}} --devices " + device,
                    )


if __name__ == "__main__":
    unittest.main()
