# Third-party components

This private research snapshot retains the following dependencies used by the trained model:

- `third_party/tiger_models`: TIGER model code from the DAVE research checkout. The retained Apache 2.0 license is in `third_party/TIGER_LICENSE`. Original source headers are preserved.
- `third_party/cmlr/espnet`: the CMLR encoder's ESPnet implementation. Original source headers are preserved, and the supplied upstream license is in `third_party/CMLR_LICENSE`. Its research-use provisions remain applicable. `config.json` contains architecture settings only.
- `third_party/ipdnet2_upstream`: the IPDNet2 architecture snapshot used in training. No standalone license file was present in the retained upstream checkout; public redistribution terms remain unverified.
- `third_party/ipdnet2_av_exp3` and `third_party/ipdnet2_ft`: circular-six visual conditioning and preprocessing adaptations. Training-only dataset loading and machine-specific paths were removed for this inference release without changing the inference computation.
- Mamba is installed separately and is not bundled as a binary or source dependency in this repository.

The five demo scenes originate from the internal SAVE synthetic test release. They contain source-derived audiovisual material and are included for private research review. Their inclusion does not establish permission for public redistribution. No license for the complete repository or media is implied by any individual dependency's license.

Public release requires a separate review of the IPDNet2 and source-media redistribution terms. Checkpoints and the full dataset are excluded.
