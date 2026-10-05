# Model and code provenance

- NanoJev: https://github.com/TianyuCodings/NanoJev (MIT). Source revision and model
  snapshot are pinned in `ai/models.json`. The Mac adapter imports the upstream
  DecisionModel, tokenizer path preparation and probability decoder. Cached-head
  training follows that head computation; no generative substitute is labeled NanoJev.
- NanoJev model: https://huggingface.co/C-Tianyu/NanoJev . The released checkpoint
  was trained for games, not this Korean robot-command task. The small Raptor head
  is a separate task adaptation, not an upstream release.
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B (Apache-2.0), distributed here
  through Ollama's `qwen3:0.6b` quantized package.
- Laya: https://github.com/NandhaKishorM/laya (Apache-2.0, code and weights),
  optional backend `ai/laya_backend.py`; checkpoint revision pinned in `ai/models.json`.
  Weights are not bundled. See `docs/LAYA_REVIEW.ko.md`.
- Ollama: https://github.com/ollama/ollama (MIT).
- Blender meshes are generated locally by `modeling/build_visuals.py`. Reference
  imagery informed the appearance only; no dimensional or fabrication claim follows
  from those images. The original reference pictures are not bundled in the repo.

Downloaded original weights are excluded from Git. HF private backup assets retain
their original metadata/licenses where supplied. The NanoJev model card did not
declare a model license at inspection; the upstream code MIT license must not be
assumed to license every checkpoint asset. The upstream code license is retained in the
source archive and `vendor/NanoJev/LICENSE` after restoration.

The Raptor project itself is licensed under PolyForm Noncommercial 1.0.0 (`LICENSE`).
That license does not apply to the third-party code, models or weights listed above;
each keeps its own original license.
