# Third-party notices

The release preserves upstream source headers and separates code, pretrained weights and source-data provenance.

- **DAVE/TIGER:** retained research source from the DAVE checkout. Its Apache 2.0 notice is in `third_party/paper_runtime/dave/LICENSE`. The adapted checkpoint is identified separately in `PROVENANCE.json`.
- **TalkNet:** audiovisual model sources and the frozen TalkSet image checkpoint. The retained upstream code notice is in `third_party/paper_runtime/talknet/LICENSE.md`. Frozen image weights are distinct from the adapted temporal components.
- **IPDNet2:** architecture from Audio-WestlakeU/FN-SSL, with circular-six and visual-conditioning adaptations. The upstream README at commit `76fcb281be92caf068c712dfb015e354f437260f` declares MIT; that retained declaration is in `docs/notices/IPDNet2_upstream_README.md`. It does not independently establish checkpoint-specific redistribution terms.
- **Mamba:** installed separately; no platform-specific extension binary is included in the public implementation archive.

No license of a component is asserted to cover all trained weights, source speech, mouth crops, noise recordings or room-response assets. Consult the relevant upstream terms. No CMLR encoder code or weights are included in the active runtime or model bundle.
