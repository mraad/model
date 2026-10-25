# Fork notes: error fixes on top of Clay-foundation/model

This fork ([mraad/model](https://github.com/mraad/model), branch `fix/errors`) is
upstream [Clay-foundation/model](https://github.com/Clay-foundation/model) `main` @
`f14e698` plus the eight error fixes below. Each fix is one commit. Each problem was
reproduced on upstream before it was changed, and `tests/test_fixes.py` covers every fix.

The model, its weights and `configs/metadata.yaml` are unchanged. With
`checkpoints/clay-v1.5.ckpt`, the classify, segment, regression and embedder factories
load the same encoder weights as upstream (identical SHA-256 of all 265 tensors), and
clay-explorer's embeddings of a test scene come out bit-identical.

```bash
.venv/bin/python -m unittest discover -s tests -v   # 10 tests, CPU, ~10 s
```

The weight-loading test that needs the 4.9 GB checkpoint skips when the checkpoint is absent.

## Fixes

| # | Area | Upstream behaviour | Fix | Commit |
|---|---|---|---|---|
| E1 | `configs/config.yaml` | Callback `class_path: src.callbacks_wandb...` points at the pre-rename module, so `trainer.py fit --config configs/config.yaml` fails while parsing the config (exit 2). | `claymodel.callbacks_wandb...` (also the commented example in `regression_biomasters.yaml`). | `7667977` |
| E2 | `ClayDataModule` predict | `setup("predict")` passes `platform=`/`metadata_path=`, which `EODataset` does not take, and reads `self.platform`, which does not exist: `AttributeError`. The predict loader also skipped `batch_collate` and could mix platforms in a batch. | Predict dataset built like fit, one `.npz` (one platform) per batch through `batch_collate`, no random time/latlon zeroing (`EODataset(zero_meta_prob=0.0)`; the default stays 0.2). | `0af71d4` |
| E3 | Encoder weight loading (4 factories) | Classify/embedder print "No matching parameter" and keep a random encoder (265 of 265 for a wrong checkpoint, exit 0). Segment/regression print nothing and leave the random encoder **trainable**. | One shared `claymodel.model.load_encoder_weights` raises `ValueError` when any encoder weight is missing or has another shape. Net −50 lines. | `768eb1f` |
| E4 | `claymodel/__init__.py` | `__all__` names objects that are never imported: `from claymodel import *` raises `AttributeError`. | `__all__` removed (importing them would load torch/lightning/timm on `import claymodel`). | `95e32f5` |
| E5 | `claymodel.finetune`, `datamodule.py` | Re-exports wrapped in `try/except ImportError`. `torchgeo` and `segmentation-models-pytorch` are used but not declared, so after `pip install` the import fails silently and `claymodel.finetune.__all__ == []`. | Re-export blocks removed (nothing used them), so a missing package is reported by the module that needs it. New extra: `pip install "claymodel[finetune]"`. | `cb1c049` |
| E6 | `utils/check_data_sanity.py` | Runs on import against `/fsx`, deletes a file on *any* exception, and deletes every file without 128 chips, including all `split_npz.py` output (32 chips each). | CLI: `check_data_sanity.py DIR [--chips N] [--delete]`. Dry run by default. Deletes only unreadable files, files without `pixels`, and (with `--chips`) files with the wrong chip count. | `4fa7ee6` |
| E7 | `LogIntermediatePredictions` | 6 figures per validation, only the last closed: 5 leak per validation over a long run. | `plt.close(fig)` inside the loop. | `ae87489` |
| E8 | `ClayMAE.forward` | Random channel dropping (all channels for 10% of chips, half for 20%) also runs in eval, so `val/loss` (which selects checkpoints) is noisy and measured on inputs inference never sees. | Applied only when `self.training`. | `141ac1c` |

## Behaviour changes to know about

- **Wrong checkpoints now fail.** Code that relied on the loaders accepting a checkpoint
  without `model.encoder.*` weights now gets a `ValueError`. Before, it silently used a
  random encoder.
- **Predict batches** are one `.npz` file each, so `batch_size` no longer applies to
  `predict_dataloader`. Predict output is also repeatable now.
- **`val/loss` is not comparable** with upstream training logs (E8). Training itself is unchanged.
- **`check_data_sanity.py`** needs a directory argument and only deletes with `--delete`.
- **Classify/segment** need `claymodel[finetune]`. Upstream had the same need but didn't declare it.

## Reported, not changed

Recorded in `tasks/todo.md`, which holds the audit of 2026-10-23:

- `module.py`: `eta_min = lr * 100`, so the cosine schedule rises to 100× the base LR.
  v1.5 was trained this way, and it's unclear whether that was intended.
- `utils.py` position encoding: frequencies collapse because of `2*omega/dim`. v1.5
  weights depend on it, so change it only for a new pretraining run.
- `datamodule.py` train/val split has no `random_state`. Every config sets
  `seed_everything`, so the split is reproducible in practice.
- Cleanup items that are not errors: import-time global settings in `model.py`, dead code
  (`ClaySampler`, `ProgressiveResizing`, `mrl.py`), the unused `vit-pytorch` dependency,
  the duplicate ruff configs, and a test for `ClayDistributedSampler`.

## Not run

No pretraining or fine-tuning run. That needs the `/fsx` data, the EuroSAT/Chesapeake/
BioMasters datasets and W&B. The fixes were checked with the unit tests above, by
parsing `configs/config.yaml` with `--print_config`, by the factory weight fingerprints,
and by clay-explorer's end-to-end embedding check (PASS for the base checkpoint and one
fine-tuned checkpoint).

## Keeping up with upstream

```bash
git fetch upstream && git rebase upstream/main   # on fix/errors
.venv/bin/python -m unittest discover -s tests
```
