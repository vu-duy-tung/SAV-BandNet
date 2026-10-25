# Fine-tuning and checkpoint provenance

The public default bundle is the TalkNet-based checkpoint whose extraction stage used 4,096 training scenes (original head SHA-256 `55505a92fcb523457790ff0d9f1aedb1aab4986eaf57f0d85c6e3b357fa09b59`). It is distinct from the later 32,768-scene model and the historical CMLR models. Their reported test scores must not be attributed to this bundle. `PROVENANCE.json` identifies its components.

The training command implements full-coverage continued fine-tuning, with fresh optimizer states. It is a consolidated reproduction recipe, not a claim of bitwise reproduction of historical training. The stages are:

| Stage | Data and supervision | Default schedule |
|---|---|---|
| Visually conditioned IPDNet2 | All TR scenes; source-location labels, activity, target spectral supervision; frozen TalkNet image encoder | 2 epochs; AdamW 1e-5 backbone / 1e-4 adapter |
| DAVE | All TR scenes; 1 s inputs with clean-source PIT and frozen 4 s teacher supervision | 1 epoch; AdamW 1e-5 |
| Feature cache | All TR/CV scenes; frozen updated localizer, separator and image features | One pass; NAS output |
| SAV selector and temporal components | Full scene sequences; central-emission reference-choice loss and clean active-speaker auxiliary loss | 4 epochs; AdamW 1e-5 |
| Speaker identity projection | Eligible TR candidates, source-speaker labels, separate scene-level internal validation | 20 epochs; AdamW 1e-3 |

Localization and separator selection use CV loss; selector selection uses normalized CV reference-choice regret. Identity fitting uses a fixed TRAIN scene holdout. The test split is never used for training or selection. Existing voice-memory gate settings remain fixed. Image weights stay frozen, and pretrained temporal batch-normalization statistics remain fixed. Candidate alignment uses overlapping waveform correlation; the learned identity projection adjusts target-selection scores, not candidate permutations.

The data manifest must contain 32,768 TR, 1,000 CV and 2,000 TT scenes. Every training scene is covered in each corresponding epoch. Windows use zero padding at session boundaries. Features and checkpoints are written beneath `--output`; use a NAS location. Completed stages can be resumed with the same output directory; incomplete stages restart. Do not change arguments or input data when resuming.

`--limit` and `--smoke-steps` are debugging controls and produce a smoke-only receipt. A smoke pass establishes executable gradients and interfaces, not convergence or final performance. Outputs contain the actual protocol, checkpoint hashes, stage receipts and progress.
