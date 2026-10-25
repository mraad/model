"""Regression tests for the fork fixes listed in FORK.md.

    .venv/bin/python -m unittest discover -s tests -v

Runs on CPU without the 4.9 GB checkpoint; tests that need it skip when it is absent.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/clay-v1.5.ckpt"


def write_chips(root, platform, bands, files=2, chips=3, size=16):
    folder = Path(root) / platform
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(files):
        np.savez(
            folder / f"{i}.npz",
            pixels=rng.random((chips, bands, size, size), dtype=np.float32),
            week_norm=np.ones((chips, 2), np.float32),
            hour_norm=np.ones((chips, 2), np.float32),
            lat_norm=np.ones((chips, 2), np.float32),
            lon_norm=np.ones((chips, 2), np.float32),
        )


class ConfigTest(unittest.TestCase):
    def test_pretraining_config_parses(self):  # E1
        run = subprocess.run(
            [sys.executable, "trainer.py", "fit", "--config", "configs/config.yaml", "--print_config"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])


class PredictTest(unittest.TestCase):
    def test_predict_batches_one_platform_without_metadata_dropout(self):  # E2
        from claymodel.datamodule import ClayDataModule

        with tempfile.TemporaryDirectory() as tmp:
            write_chips(tmp, "linz", bands=3)
            write_chips(tmp, "naip", bands=4)
            dm = ClayDataModule(
                data_dir=tmp, size=16, metadata_path=str(ROOT / "configs/metadata.yaml"),
                platforms=["linz", "naip"], batch_size=4, num_workers=0,
            )
            dm.setup("predict")
            batches = list(dm.predict_dataloader())
        self.assertEqual(len(batches), 4)  # one per file
        for batch in batches:
            self.assertEqual(len(set(batch["platform"])), 1)
            bands = 3 if batch["platform"][0] == "linz" else 4
            self.assertEqual(tuple(batch["pixels"].shape), (3, bands, 16, 16))
            self.assertTrue(bool((batch["time"] == 1).all() and (batch["latlon"] == 1).all()))


if __name__ == "__main__":
    unittest.main()
