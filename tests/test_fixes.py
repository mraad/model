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
            [
                sys.executable,
                "trainer.py",
                "fit",
                "--config",
                "configs/config.yaml",
                "--print_config",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])


class PackageImportTest(unittest.TestCase):
    def test_star_import(self):  # E4
        run = subprocess.run(
            [sys.executable, "-c", "from claymodel import *"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])

    def test_core_imports_without_finetune_extra(self):  # E5
        # None in sys.modules makes `import torchgeo` fail like a missing package.
        code = (
            "import sys; sys.modules['torchgeo'] = sys.modules['segmentation_models_pytorch'] = None\n"
            "import claymodel.datamodule, claymodel.finetune\n"
            "from claymodel.finetune.embedder.factory import Embedder\n"
            "try:\n"
            "    import claymodel.finetune.classify.eurosat_datamodule\n"
            "except ImportError as e:\n"
            "    print('missing:', e.name)\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        self.assertIn("missing: torchgeo", run.stdout)


class PredictTest(unittest.TestCase):
    def test_predict_batches_one_platform_without_metadata_dropout(self):  # E2
        from claymodel.datamodule import ClayDataModule

        with tempfile.TemporaryDirectory() as tmp:
            write_chips(tmp, "linz", bands=3)
            write_chips(tmp, "naip", bands=4)
            dm = ClayDataModule(
                data_dir=tmp,
                size=16,
                metadata_path=str(ROOT / "configs/metadata.yaml"),
                platforms=["linz", "naip"],
                batch_size=4,
                num_workers=0,
            )
            dm.setup("predict")
            batches = list(dm.predict_dataloader())
        self.assertEqual(len(batches), 4)  # one per file
        for batch in batches:
            self.assertEqual(len(set(batch["platform"])), 1)
            bands = 3 if batch["platform"][0] == "linz" else 4
            self.assertEqual(tuple(batch["pixels"].shape), (3, bands, 16, 16))
            self.assertTrue(
                bool((batch["time"] == 1).all() and (batch["latlon"] == 1).all())
            )


class CheckDataSanityTest(unittest.TestCase):
    """E6: dry run by default, keeps split_npz output, deletes only bad files."""

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, "utils/check_data_sanity.py", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def test_dry_run_then_delete_only_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_chips(tmp, "linz", bands=3, files=1, chips=32)  # split_npz.py output
            bad = Path(tmp) / "linz/corrupt.npz"
            bad.write_bytes(b"not a zip")
            np.savez(Path(tmp) / "linz/nopixels.npz", other=np.zeros(1))
            files = lambda: sorted(p.name for p in Path(tmp).rglob("*.npz"))
            everything = files()

            out = self.run_script(tmp, "--chips", "32")
            self.assertIn("2 of 3 .npz files invalid.", out)
            self.assertEqual(files(), everything)

            self.run_script(tmp, "--chips", "32", "--delete")
            self.assertEqual(files(), ["0.npz"])

    def test_import_has_no_side_effects(self):
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy; runpy.run_path('utils/check_data_sanity.py', run_name='x')",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual((run.returncode, run.stdout), (0, ""), run.stderr[-2000:])


class LoadEncoderWeightsTest(unittest.TestCase):
    """E3: a checkpoint without the encoder weights must fail, not train or embed
    with a random encoder."""

    def test_wrong_checkpoint_raises(self):
        from claymodel.finetune.embedder.factory import Embedder
        from claymodel.finetune.segment.factory import Segmentor

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "wrong.ckpt"
            torch.save({"state_dict": {"model.decoder.x": torch.zeros(1)}}, ckpt)
            with self.assertRaisesRegex(ValueError, "265 of 265 encoder parameters"):
                Embedder(img_size=16, ckpt_path=ckpt)
            with self.assertRaisesRegex(ValueError, "265 of 265 encoder parameters"):
                Segmentor(num_classes=2, ckpt_path=ckpt)

    @unittest.skipUnless(CKPT.exists(), f"{CKPT} not downloaded")
    def test_clay_checkpoint_loads_and_freezes(self):
        from claymodel.finetune.embedder.factory import Embedder

        encoder = Embedder(img_size=16, ckpt_path=CKPT).clay_encoder
        sd = torch.load(CKPT, map_location="cpu", mmap=True)["state_dict"]
        weight = encoder.transformer.layers[0][0].to_qkv.weight
        self.assertTrue(
            torch.equal(
                weight, sd["model.encoder.transformer.layers.0.0.to_qkv.weight"]
            )
        )
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
