"""Test the maintainer manifest against the reader's actual installer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
from feature_information_dynamics.data.releases import download, verify

_spec = importlib.util.spec_from_file_location("release_builder", Path(__file__).resolve().parents[1] / "tools/build_data_release_manifest.py")
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)


class ManifestBuilderTests(unittest.TestCase):
    def test_existing_historical_format_installs_without_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "source"; source.mkdir()
            for name in ["paired", "mask_shards", "canny_shards"]:
                (source / name).mkdir()
            names = ["n00000001/a.JPEG", "n00000002/b.JPEG"]
            (source / "class_ids.json").write_text(json.dumps({"n00000001": 0, "n00000002": 1}))
            for split, selected in [("train", names[:1]), ("val", names[1:])]:
                (source / "paired" / ("paired_" + split + "_whitelist.json")).write_text(json.dumps({"filenames": selected}))
            masks = np.zeros((2, 256, 256), dtype=np.uint8)
            save_file({"masks": masks, "mask_valid": np.array([1, 0], dtype=np.uint8)}, str(source / "mask_shards/masks_rank00_shard000.safetensors"))
            save_file({"cannys": masks, "canny_valid": np.ones(2, dtype=np.uint8)}, str(source / "canny_shards/cannys_rank00_shard000.safetensors"))
            (source / "canny_shards/cannys_rank00_shard000.json").write_text(json.dumps({"filenames": names}))
            manifest = source / "release.json"
            result = _builder.build(source, "synthetic-download-test", manifest)
            self.assertEqual(result["validation"]["mask_invalid_paired"], 1)
            self.assertNotIn("vavae_latents", result["resources"])
            installed = base / "installed"
            download(manifest, installed)
            checked = verify(manifest, installed)
            self.assertEqual(checked["validation"], result["validation"])
            self.assertFalse((installed / "vavae_latents").exists())

    def test_missing_conditions_rejected_before_manifest_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.json"
            with self.assertRaisesRegex(ValueError, "No prepared files"):
                _builder.build(directory, "synthetic-invalid", output)
            self.assertFalse(output.exists())
