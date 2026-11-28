# agents_results

Research subagent logs: each file records the question asked, the scope (paths, commits), the results with VERIFIED/INFERRED marks, and what could not be verified. Search here before starting new research.

| File | Topic |
|---|---|
| [2026-11-25_qwen3-omni-g0-code-audit.md](2026-11-25_qwen3-omni-g0-code-audit.md) | Qwen3-Omni video→text source audit (M* 2c8ffde8, vLLM-Omni v0.34.0): video preprocessing, encoder cache, R1′, prefix cache defaults/switches, prompt layout, 5880 deploy configs, hook points, S1/S2 threats |
| [2026-11-26_vllm-omni-streaming-input-audit.md](2026-11-26_vllm-omni-streaming-input-audit.md) | vLLM-Omni v0.34.0 streaming/incremental input: WS video stream (frames as per-frame image items, audio-output gate), duplex models, vLLM 0.34 StreamingInput, cross-session encoder/prefix cache lifetime, timing hooks and benchmarks |
| [2026-11-26_vllm-omni-streaming-input-deep-dive.md](2026-11-26_vllm-omni-streaming-input-deep-dive.md) | Deep dive on the audit's open items: audio-gate origin (PR #5894), text-only warmup verdict, image mrope prefix invariance, encoder cache 62720 and frames that fit, KV estimate, measurement hooks, corrections |
| [2026-11-26_gate-reason-and-text-warmup-hidden-states.md](2026-11-26_gate-reason-and-text-warmup-hidden-states.md) | vLLM-Omni audio-output gate: reason, tests covering audio + prefix hit, text-only warmup routing and hidden-state save path, verdict and GPU test design |
| [2026-11-26_tmrope-image-items-prefix-invariance.md](2026-11-26_tmrope-image-items-prefix-invariance.md) | Qwen3-Omni image-item mrope positions: prefix invariance (CPU check), image vs video t-index, use_audio_in_video on images, 640x480 and 1280x720 token counts |
| [2026-11-26_encoder-cache-capacity-streaming-frames.md](2026-11-26_encoder-cache-capacity-streaming-frames.md) | vLLM-Omni stage-0 encoder cache size (62720 embeddings), frames that fit per resolution, eviction and warmup-abort behaviour, thinker KV capacity estimate |
| [2026-11-26_measurement-hooks-process-cache-timing.md](2026-11-26_measurement-hooks-process-cache-timing.md) | Measurement hooks: process model and sitecustomize, num_cached_tokens path per runner, warmup abort block reuse, ModelState encoder hook, mm-processor-stats wiring |
| [2026-11-26_mstar-streaming-input-audit.md](2026-11-26_mstar-streaming-input-audit.md) | M* 2c8ffde8f streaming and reuse audit (Qwen3-Omni): streaming paths P1–P8, Qwen3-Omni positions and exactness (not exact; upload-time prefix option), encoder-output cache insertion points, measurement wrappers on perf_counter |
| [2026-11-26_mstar-streaming-inventory.md](2026-11-26_mstar-streaming-inventory.md) | M* streaming and session paths: per-path models, client protocol, per-chunk vs commit work, KV persistence |
| [2026-11-26_mstar-qwen3-incremental-exactness.md](2026-11-26_mstar-qwen3-incremental-exactness.md) | M* Qwen3-Omni incremental prefill: TMRoPE dependence on video duration, audio interleave limits, walk runnability, smallest change, exactness verdict |
| [2026-11-26_mstar-encoder-output-cache-hooks.md](2026-11-26_mstar-encoder-output-cache-hooks.md) | M* encoder-output cache: engine insertion points (prepare_inputs, exec_and_postprocess), replay vs skip, deepstack bundle size, generic component reach, risks |
| [2026-11-26_mstar-measurement-hooks.md](2026-11-26_mstar-measurement-hooks.md) | M* measurement: aggregate metrics, request ids in Engine.exec, log-stats profile, perf_counter clock across processes, wrapper set |
