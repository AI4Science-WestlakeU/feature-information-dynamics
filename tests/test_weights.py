"""Missing parents and interrupted downloads must not start a formal run."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from feature_information_dynamics.weights import check_training_checkpoint, download


class WeightPreparationTests(unittest.TestCase):
    def test_wrong_checksum_leaves_no_checkpoint(self):
        response = io.BytesIO(b"wrong version")
        response.headers = {}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.pt"
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download("https://example.org/model.pt", destination, "0" * 64)
            self.assertFalse(destination.exists())

    def test_missing_parent_rejected_and_existing_parent_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "train.yaml"
            parent = root / "parent.pt"
            config.write_text(f'train:\n  weight_init: "{parent.as_posix()}"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                check_training_checkpoint(["--config", str(config)])
            parent.write_bytes(b"checkpoint")
            check_training_checkpoint(["--config", str(config)])

    def test_incomplete_response_leaves_no_checkpoint(self):
        response = io.BytesIO(b"short")
        response.headers = {"Content-Length": "100"}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.pt"
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download("https://example.org/model.pt", destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("model.pt.part").exists())

    def test_complete_download_preserves_existing_checkpoint(self):
        response = io.BytesIO(b"checkpoint")
        response.headers = {"Content-Length": "10"}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.pt"
            with patch("urllib.request.urlopen", return_value=response):
                download("https://example.org/model.pt", destination)
            self.assertEqual(destination.read_bytes(), b"checkpoint")
            with self.assertRaises(FileExistsError):
                download("https://example.org/other.pt", destination)


if __name__ == "__main__":
    unittest.main()
