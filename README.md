# SAV-BandNet

SAV-BandNet combines spatial and visual conditioning with speaker memory for consistent target-speech extraction from six-channel mixtures. It processes 1 s windows and emits 0.4 s per step.

Install the Python/CUDA dependencies in [docs/setup.md](docs/setup.md), download the [SAVE dataset](https://huggingface.co/datasets/playwithmino/SAVE) to `$SAVE`, and download the [model bundle](https://huggingface.co/playwithmino/SAV-BandNet) to `assets/default`.

Fine-tune visually conditioned IPDNet2, DAVE, the target selector and speaker identity projection on all 32,768 training scenes:

```bash
python -m sav_bandnet train --data "$SAVE" --output "$NAS/training/sav-bandnet"
```

Infer on all 2,000 test scenes:

```bash
python -m sav_bandnet infer --data "$SAVE" --output "$NAS/evaluation/sav-bandnet"
```

The recipe uses validation data only for selection. See [docs/recipe.md](docs/recipe.md) for stage settings and [third_party/README.md](third_party/README.md) for preserved baseline artifacts.
