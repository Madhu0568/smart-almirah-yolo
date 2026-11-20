import tempfile
import unittest
from pathlib import Path

from train import load_dataset_config


class TrainingConfigTests(unittest.TestCase):
    def test_valid_dataset_yaml_is_loaded(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "clothing.yaml"
            data_file.write_text(
                "train: images/train\nval: images/val\nnames:\n  0: shirt\n  1: trousers\n",
                encoding="utf-8",
            )
            path, config = load_dataset_config(data_file)
            self.assertEqual(path, data_file.resolve())
            self.assertEqual(config["names"][0], "shirt")

    def test_missing_dataset_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset_config("this-dataset-file-does-not-exist.yaml")

    def test_dataset_yaml_requires_train_and_validation(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "invalid.yaml"
            data_file.write_text("train: images/train\nnames: [shirt]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "val"):
                load_dataset_config(data_file)

    def test_dataset_yaml_requires_class_names_or_count(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "invalid.yaml"
            data_file.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "names"):
                load_dataset_config(data_file)


if __name__ == "__main__":
    unittest.main()
