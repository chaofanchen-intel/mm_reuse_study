# Gate reason and text-only warmup hidden states (vLLM-Omni v0.34.0)

## Question(s) asked

- Q1. Why does `"audio" not in config.modalities` gate incremental prefill? Report the stated reason verbatim, or say none was stated.
- Q2. Is "audio-output + stage-0 prefix-cache hit is safe" backed by a test? List the tests that send audio output AND get a prefix hit.
- Q3. For text-only (`["text"]`) on the default Qwen3-Omni deploy (async_chunk true), is stage 1 submitted or skipped? Give the branch condition.
- Q4. Are full-prompt thinker hidden states saved to the omni prefix-cache pool regardless of output modality? Is the save gated per request or by model policy? Which runner applies?
- Q5. Verdict: would text-only warmups (`max_tokens=1`, `output_modalities=["text"]`) followed by an audio-output query that hits the frame prefix work without changing the talker? Condition, GPU check, and how the talker gets hidden states on a hit.

## Scope

- Repo: `/home/chaofan/workspaces/mm_reuse_study/external/vllm-omni`, HEAD `06a28821a7f72bdb436c3df656b10f67fded0846` (tag v0.34.0). Read only.
- Commits: `eb1699bfc` (introduces the gate, "Support incremental prefill for realtime video streaming (#5894)"); `06a28821a` (v0.34 backport, #10902, re-adds the same hunk). `git log -S'"audio" not in config.modalities'` returns only these two.
- Paths read: `vllm_omni/entrypoints/openai/video_stream_base.py`, `vllm_omni/entrypoints/openai/serving_chat.py`, `vllm_omni/entrypoints/async_omni.py`, `vllm_omni/entrypoints/omni_base.py`, `vllm_omni/entrypoints/utils.py`, `vllm_omni/engine/orchestrator.py`, `vllm_omni/engine/async_omni_engine.py`, `vllm_omni/engine/async_engine_utils.py`, `vllm_omni/engine/arg_utils.py`, `vllm_omni/config/stage_config.py`, `vllm_omni/core/sched/omni_ar_scheduler.py`, `vllm_omni/core/sched/omni_scheduler_mixin.py`, `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`, `vllm_omni/worker_v2/*` (`omni_ar_model_runner.py`, `omni_model_runner.py`, `omni_data_plane.py`, `prefix_cache.py`, `model_states/omni_model_state.py`, `model_states/mtp_model_state.py`), `vllm_omni/core/prefix_cache/*` (`interface.py`, `runner_mixin.py`, `manager.py`), `vllm_omni/model_executor/models/qwen3_omni/{pipeline.py,qwen3_omni.py}`, `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`, `vllm_omni/platforms/npu/models/qwen3_omni_model_state.py`, `vllm_omni/deploy/qwen3_omni_moe.yaml`, `docs/serving/video_stream_api.md`, `docs/design/feature/prefix_caching.md`, `tests/e2e/online_serving/test_qwen3_omni.py`, `tests/entrypoints/openai_api/test_serving_video_stream.py`, `tests/engine/kv_offload_*`, `tests/engine/test_mooncake_*`, `tests/core/test_prefix_cache*.py`, `tests/worker_v2/test_prefix_cache.py`, `tests/helpers/stage_config.py`.
- vLLM 0.34 site-packages (read only): `vllm/v1/core/kv_cache_utils.py:633-660` (block-hash extra keys).
- External, read only: public GitHub API for PR #5894 (body and comments). `gh` is not logged in, so the public REST endpoint was used without credentials.
- Nothing was executed: no Python, no GPU, no servers. All marks come from reading code, tests, docs, and PR text.

## Findings

### Q1. Why the gate exists: none stated

- The gate: `video_stream_base.py:297-299` (`_incremental_prefill_active`, returns `self._incremental_prefill_supported and "audio" not in config.modalities`). VERIFIED.
- Introduced by `eb1699bfc` ("[FEAT][Fronted] Support incremental prefill for realtime video streaming (#5894)"). The commit body has only Signed-off-by and Co-authored-by lines, no rationale. VERIFIED (`git show eb1699bfc`).
- Re-added in the v0.34 backport `06a28821a` (#10902), same hunk. VERIFIED (`git show 06a28821a`).
- Code comment at the gate, `video_stream_base.py:298`: `"""Incremental frame prefill: stage-0 prefix caching on, text-only output."""`. This describes the condition only. No reason. VERIFIED.
- Docs, `docs/serving/video_stream_api.md:117-118`: "Server-side only (stage-0 `enable_prefix_caching: true` and `modalities: ["text"]`). There is no `session.config` flag. Audio-output sessions stay on the legacy path." No reason. VERIFIED.
- Docs, `docs/serving/video_stream_api.md:147-148`: "Stage-0 prefix caching is safe for audio-output requests: on a hit, the omni prefix cache restores the thinker hidden states the talker needs. A preempted request that resumes on a hit does not get them back, so size the deployment to avoid preemption." VERIFIED.
- PR #5894 body (public API, read only), verbatim: "This is request-level stage-0 prefix-cache warmup for text-only sessions. It turns on when stage-0 `enable_prefix_caching` is true and `modalities` does not include `audio`. There is no `session.config` flag. Audio-output sessions on this endpoint stay on the legacy path and do not start warmup." The body also says: "Stage-0 prefix caching can serve audio-output requests: a hit restores the thinker hidden states the talker needs." No technical reason is given for excluding audio output. VERIFIED (GitHub API, not the repo).
- PR #5894 comments: none on the gate. The review comment from another contributor concerns legacy-path frame waiting (`video_stream_base.py:565-569`), not the audio gate. VERIFIED.
- Test `tests/entrypoints/openai_api/test_serving_video_stream.py:1552-1566` (`test_audio_session_never_starts_warmup`) asserts zero warmups for an audio session. No reason in the test. VERIFIED.
- Possible cost, not stated anywhere: the warmup passes `config.modalities` to the engine (`video_stream_base.py:1392` in the request kwargs and `:1412` to `_drive_warmup_generate`). An audio session would therefore warm with audio output, which would drive talker and code2wav for a `max_tokens=1` request. INFERRED. This matches the earlier audit `agents_results/2026-11-26_vllm-omni-streaming-input-audit.md`, which reached the same point.

Stated reason: none. The PR and docs define the scope ("text-only sessions", "stay on the legacy path") and never say why audio output is excluded.

### Q2. Tests combining audio output with a stage-0 prefix hit

Tests that send audio output AND get a stage-0 prefix hit (in-stage omni prefix cache, Qwen3-Omni):

1. `tests/e2e/online_serving/test_qwen3_omni.py:263-285` `test_thinker_prefix_caching_audio_output`. Default modalities (absent field resolves to engine modalities `["text","audio"]`, `serving_chat.py:822-824`). Two identical streaming requests. Asserts the second request has `cached_tokens > 0` and block-aligned (`_run_prefix_cache_check`, `:189-223`). Does NOT assert the audio payload. Server `--no-async-chunk` and prefix caching on stages 0 and 1 (`:40-45`). H100 x2 (`:259-260`). Meets the combination, but the audio content is not checked. Whether it passes: UNVERIFIED (not run).
2. `tests/e2e/online_serving/test_qwen3_omni.py:372-395` `test_thinker_prefix_caching_async_chunk_identical`. Async chunk (`:63-74`), default modalities, two identical requests. Asserts a block-aligned hit AND audio bytes on both responses (`_assert_omni_payload_complete`, `:358-364`). This is the strongest combination found, but the prompt is identical, so it is a full-prompt replay, not a partial prefix hit. Whether it passes: UNVERIFIED.

Adjacent, different mechanism (not the in-stage prefix cache):
3. `tests/engine/test_kv_offload_consistency_qwen3.py` with `tests/engine/kv_offload_helpers.py:150-181`. LMCache KV plus hidden-state offload. A cache hit must reproduce the no-cache baseline audio waveform. Requires lmcache (`test_kv_offload_consistency_qwen3.py:53`, `pytest.importorskip`). The helper sets `enable_prefix_caching: False` (`kv_offload_helpers.py:79`), so it does not exercise the in-stage prefix cache. VERIFIED (read).
4. `tests/engine/test_mooncake_turn_consistency_qwen3.py`, `tests/engine/test_mooncake_duplex_consistency_qwen3.py`. Opt-in, need `MOONCAKE_CONFIG_PATH` and a running Store (header). Cross-turn restore of talker inputs, not the in-stage cache. Run status: UNVERIFIED.

Does NOT count:
- Shared-prefix audio-free test: `test_qwen3_omni.py:398-` `test_thinker_prefix_caching_async_chunk_shared_image_prefix` is a partial-prefix hit, but `"modalities": ["text"]` at `:422`. Text only.
- `test_thinker_prefix_caching_text_output` (`:226`), `test_batched_completions_text` (`:296`): text only.
- Unit tests `tests/core/test_prefix_cache.py:489-680` (synthetic `talker.h` and `codes.audio` keys), `tests/worker_v2/test_prefix_cache.py`, `tests/core/test_prefix_cache_runner_mixin.py`: mock or synthetic, no real audio request.
- `test_serving_video_stream.py:1552` (audio never warms) and `:1766` (`test_reuse_query_with_audio_matches_legacy_prompt`, prompt shape only): mocked engine, no audio output from the engine, no hit.
- PR #5894 test plan: GPU A/B on 2x H20 with text-only sessions only ("Client: text-only sessions"). No audio-output run. VERIFIED (PR body).

No test found for: a text-only warmup followed by an audio-output query; an audio-output query with a partial-prefix hit (num_cached < prompt, with audio); or an audio-content comparison of a hit against a cold run in the in-stage cache.

Stale comment: `test_qwen3_omni.py:33-38` says "The production YAML leaves the Thinker prefix cache off with model runner v2." The deploy YAML sets stage 0 `enable_prefix_caching: true` (`vllm_omni/deploy/qwen3_omni_moe.yaml:40`) and turns it off only in the rocm, npu, musa, and xpu platform sections (`:108`, `:119`, `:146`, `:153`). The CI overlay (`tests/helpers/stage_config.py:371-395`) does not change prefix caching. VERIFIED (read). The comment is contradicted by the YAML on CUDA.

### Q3. Stage routing for text-only vs audio

1. Chat serving: `request.modalities` or the engine default (`serving_chat.py:822-824`), passed as `output_modalities` (`serving_chat.py:1050`). VERIFIED.
2. Pipeline flags: stage 0 `final_output=True, final_output_type="text"` (`vllm_omni/model_executor/models/qwen3_omni/pipeline.py:40-41`); stage 1 `input_sources=(0,)` with no `final_output` (default False, `vllm_omni/config/stage_config.py:406`); stage 2 `final_output=True, final_output_type="audio"` (`pipeline.py:86-87`). VERIFIED.
3. Final-stage selection: `get_final_stage_id_for_e2e` (`vllm_omni/entrypoints/utils.py:181-216`) scans from the last stage down and picks the first with `final_output` and `final_output_type` in the requested modalities. `["text"]` gives 0. `["text","audio"]` gives 2. VERIFIED by reading (not executed).
4. `async_omni.py:293-294` computes `final_stage_id_for_e2e` and `final_output_stage_ids`; `:308` passes `final_stage_id` into the request. `async_omni_engine.py:402-403` stamps `omni_final_stage_id` on the request (`async_engine_utils.py:236-256`). VERIFIED.
5. Prewarm (async chunk): `_prewarm_async_chunk_stages` (`vllm_omni/engine/orchestrator.py:3660`) returns early at `:3679-3681` when `req_state.final_stage_id <= entry_stage_id`. Otherwise it loops `range(entry_stage_id + 1, req_state.final_stage_id + 1)` (`:3712`). For text-only the loop is empty and no stage-1 placeholder is submitted. Callers at `:4384` and `:4425`. VERIFIED.
6. Forward: `_route_output` (`orchestrator.py:2440`). The forward to the next stage requires `stage_id < req_state.final_stage_id` (`:2532`, within the condition at `:2529-2535`). For text-only, `0 < 0` is false, so no forward. Stage 0's output goes to the frontend as the final output (`:2486-2491`). VERIFIED.
7. Chunk adapter: `chunk_transfer_adapter.py` has no modality logic (grep found none). On the default MRv2 deploy, stage 0 has `supports_native_mrv2_data_plane=True` (`pipeline.py:52`), so `_native_data_plane` is True and `chunk_transfer_adapter` is None for that stage (`omni_scheduler_mixin.py:376-383`). The V1-style omission for `omni_final_stage_id == 0` (`omni_ar_scheduler.py:443-450`, `:1634-1640`, `:1807-1818`) therefore does not run for stage 0 on the default deploy. VERIFIED (condition read), INFERRED for runtime effect.
8. Native data plane: `worker_v2/omni_ar_model_runner.py:1215-1255` publishes `inter_stage_outputs` whenever `_async_chunk` is on, with no final-stage check. `omni_data_plane.py:147-151` sets `_can_send` True when `custom_process_func` is set, and `publish_output` enqueues the payload (`:227-242`). Grep finds no `omni_final_stage_id` check in `worker_v2` except the NPU model state. INFERRED: for text-only, stage 0 still computes and enqueues the talker-bound payload, but no stage-1 request consumes it. Whether it is released at finish: UNVERIFIED.

Answer: for text-only, stage 1 is NOT submitted (no prewarm, no forward, no stage-1 request). The controlling conditions are `final_stage_id` from `get_final_stage_id_for_e2e` and `stage_id < req_state.final_stage_id` (`orchestrator.py:2532`) plus the prewarm early return (`:3680`). Stage 0 still produces the talker-bound payload on the native plane (INFERRED).

### Q4. Save path for stage-0 hidden states

Runner: MRv2 (`worker_v2`) is the default and the only reachable path in this tree.
- Deploy: `model_runner: v2` (`vllm_omni/deploy/qwen3_omni_moe.yaml:22`); pipeline default `model_runner: Literal["v1","v2"] = "v2"` (`vllm_omni/config/stage_config.py:932`); `use_v2_model_runner: bool = True` (`vllm_omni/engine/arg_utils.py:346`); V1 raises at `arg_utils.py:484-485` ("Model Runner V1 is not supported with vLLM 0.34; select model_runner: v2"). VERIFIED.
- Stage-0 prefix cache: on for CUDA (`qwen3_omni_moe.yaml:40`), off for rocm, npu, musa, xpu (`:108`, `:119`, `:146`, `:153`). Doc `docs/design/feature/prefix_caching.md:256-257`: thinker supported on MRv2; talker native-KV only. VERIFIED.

Save call chain (MRv2):
- `worker_v2/omni_ar_model_runner.py:307`: `prefix_cache_rewrite = self._prefix_cache_save_mrv2(input_batch, text_hidden, multimodal_outputs)`. VERIFIED.
- `worker_v2/prefix_cache.py:167-214` (`_prefix_cache_save_mrv2`) calls `_prefix_cache_save_step`, keeps a rewrite only when `rewrite_ids` exist and `engine_output_type != "text"` (`:193-205`). VERIFIED.
- `core/prefix_cache/runner_mixin.py:168-222` (`_prefix_cache_save_step`): returns None only when the cache is absent, the model is pooling, the rank is not the last PP rank, or a pure-decode step is buffered. Otherwise `save_outputs(hidden, flatten(mm), ...)`. It does NOT read `additional_information`, `output_modalities`, or `omni_final_stage_id`. VERIFIED.
- So the save is gated by the model policy and cache config only, not by a per-request flag. VERIFIED.

Policy:
- MRv2 caches outputs only when the model sets `supports_mrv2_prefix_cache_outputs` (`worker_v2/prefix_cache.py:118-124`; otherwise native KV only). Qwen3-Omni's thinker sets it True at `qwen3_omni.py:297`, inside the `if self.model_stage == "thinker":` branch that starts at `:272` (indentation). VERIFIED.
- Default `needs_full_hidden_states=True` (`core/prefix_cache/interface.py`, `ModelCachePolicy`, `from_model` around `:185-193`). VERIFIED.
- Thinker output has no modality branch: `make_omni_output` returns captured layers and `embed` for the thinker regardless of output modalities (`qwen3_omni.py:886-916`). VERIFIED.
- Postprocess: default `worker_v2/model_states/omni_model_state.py:1152-1196` has no final-stage check. VERIFIED by reading.
- The only text-only short-circuit is NPU-specific: `platforms/npu/models/qwen3_omni_model_state.py:28-50` drops multimodal outputs for text-only batches only when `enable_prefix_caching` is OFF. With caching on, outputs are kept. VERIFIED. This suggests text-only outputs are meant to exist when caching is on. CUDA uses the default state (INFERRED).
- Mergeability: the thinker is not a `RequestOutputSnapshot` producer (producers: fish_omni, moss_tts, ming3_tts, higgs_audio_v3 only, grep). Streaming audio is None unless `_audio_buffer` is set (`mtp_model_state.py:318-326`, buffer created conditionally at `:556`). So the rewrite is allowed (`omni_ar_model_runner.py:413-417`, `mergeable=...`). INFERRED for the thinker's `_audio_buffer` being None.

Answer: YES. Full-prompt thinker hidden states and captured layers are saved for any stage-0 prefill regardless of output modality, gated by model policy and cache config. Applies to the MRv2 path (`worker_v2`). The V1 path (`vllm_omni/worker`) is unreachable in this tree and was not analyzed.

### Q5. Verdict

Verdict: LIKELY WORKS at the engine level (INFERRED from code). Not verified on GPU. The gate itself is not an engine blocker; the change needed is in the handler, not the talker.

Chain:
1. Warmup (text-only, `max_tokens=1`): final stage 0, so no talker or code2wav submission (`orchestrator.py:3680`, `:2532`). The thinker saves hidden states and captured layers for all prompt tokens (`omni_ar_model_runner.py:307`; `worker_v2/prefix_cache.py:167-214`; `runner_mixin.py:168-222`). The save does not depend on modality. VERIFIED (path); INFERRED that the saved rows are complete for the frame prefix.
2. Query (audio, final stage 2): talker and code2wav are prewarmed (`orchestrator.py:3712`). vLLM's block hash is built from tokens, mm hashes, LoRA, cache salt, and prompt embeds (`vllm/v1/core/kv_cache_utils.py:633-660`), with no output-modality term. So the query can hit the warmup's blocks if the prompt tokens, mm hashes, and processor kwargs (`use_audio_in_video`, `video_stream_base.py:951-955`) match. VERIFIED.
3. On a hit, `PrefixCacheRewrite` (`worker_v2/prefix_cache.py:39-88`) replaces `"hidden"` and each flattened mm key present in the payloads with full-prompt rows from the pool (`materialize(..., include_uncached=False)`). Applied through `_prefix_cache_output_finalizer` (`prefix_cache.py:217-240`, called from `omni_ar_model_runner.py:413-417`). Payloads are flattened (`omni_ar_model_runner.py:695`, `data_entry_keys.py:434-440`), so the flattened `hidden_states` layer keys read by the talker processor line up with the pool keys. VERIFIED for each step; naming alignment INFERRED.
4. Talker hidden states: yes, the talker needs the prefix rows. `thinker2talker_async_chunk` (`stage_input_processors/qwen3_omni.py:503-530`, `_layer_tensor` at `:49`, embedding-layer index `:65`) builds the talker input from the thinker's embedding and accept-hidden layers. The talker does not read its own cache: stage 1 `enable_prefix_caching: false` (`qwen3_omni_moe.yaml:52`) and native KV only under MRv2 (`prefix_caching.md:257`). On a hit, the prefix rows come from the thinker payload restored by the rewrite (`prefix_caching.md` table row at `:256`: "full-prompt hidden states are merged from the pool on a hit"; merge description at `prefix_caching.md:135-137`).

Conditions the verdict depends on:
- (a) The handler warms with `output_modalities=["text"]`. Currently `video_stream_base.py:1392` and `:1412` pass `config.modalities`. This is a handler change, not a talker change.
- (b) Identical prompt prefix, mm hashes, and processor kwargs between warmup and query. Modality is not part of the hash, so modality alone does not break reuse.
- (c) The hit request is not preempted. Docs warn a preempted hit "does not get them back" (`video_stream_api.md:147-150`). The thinker has no `recompute_preemption="fail"`; only the talker does (`pipeline.py:79`). Under load this could be silent.
- (d) The thinker output is mergeable (INFERRED yes; see Q4).
- (e) Blocks still resident at query time. Eviction is safe (the query just misses).

Silent failure mode: if the rewrite does not replace some key the talker reads (naming mismatch or a key not pooled), the talker gets suffix-only rows for the prefix. Text would look correct; audio would degrade. This is why the GPU test must check audio, not only cached-token counts.

Most important GPU test (single check): text-only warmup (`max_tokens=1`, `modalities=["text"]`) on the frame prefix, then an audio-output query whose prompt shares that prefix with a partial hit (`num_cached_tokens > 0`, block-aligned, `< prompt_tokens`). Compare the query's talker-bound conditioning and output against a cold run of the same request (prefix caching off, or a fresh server with no warmup). Use deterministic sampling: talker seed 42 is set (`qwen3_omni_moe.yaml:66`), thinker greedy. Compare the thinker payload rows for the prompt (or the codec tokens) bit-for-bit or with a tight tolerance, not only text. Also confirm the rewrite happened (`full_payload_prefix_cache_keys` stamped with `hidden` and the layer keys, `prefix_cache.py:83-88`) and that no `OmniPrefixCacheUnmatchError` is raised. Note `test_qwen3_omni.py:232-236` says greedy text can differ across scheduling shapes, so compare the conditioning, not only the text.

## Could not verify

- No GPU run. Nothing executed. Every runtime claim (rewrite on the warm-then-audio path, talker output equality) is UNVERIFIED.
- Whether the e2e tests at `test_qwen3_omni.py:263` and `:372` pass in CI (H100 x2 hardware marks; not run).
- Whether native-plane payloads published for text-only requests (Q3 item 8) are released at finish. Release hook exists (`worker_v2/omni_model_runner.py:406-408`); not traced.
- Key-name alignment between flattened pool keys and the processor's layer keys (Q5 step 3). INFERRED.
- Whether the thinker's `_audio_buffer` is None on Qwen3-Omni (`mtp_model_state.py:556` condition not traced). INFERRED.
- The vLLM `ChatCompletionRequest.modalities` default was not checked; `serving_chat.py:822-824` handles both absent and None, so the default is still the engine modalities.
- The V1 runner path (`vllm_omni/worker`) was not analyzed; it is unreachable here per `arg_utils.py:484-485`.
- The Mooncake and LMCache tests were not run; their run status is unknown.

## Verdict on Q5

LIKELY WORKS at the engine level, UNVERIFIED on GPU. The gate is a handler-level scope decision with no stated reason. The engine-level path (warmup save, hash without modality, rewrite on hit, talker fed from the thinker payload) appears to support a text-only warmup followed by an audio query. The handler must warm with `["text"]`. The single most important GPU check is audio conditioning equivalence between a warm partial-prefix hit and a cold run, with the rewrite confirmed by metadata.

## Notes for the caller

- The README index in `agents_results/` was not updated, per the instruction to edit no other files. A row for this file can be added if wanted.
