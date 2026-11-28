# vLLM-Omni streaming video input: deep dive on the open items

## Question asked

Follow-up to [2026-11-26_vllm-omni-streaming-input-audit.md](2026-11-26_vllm-omni-streaming-input-audit.md). Its "could not verify" list left several items open. The request: verify them properly and produce an in-depth report, using sub-agents as needed.

Scope and constraints (same as the audit): read-only on `external/vllm-omni` and the vLLM 0.34 install; no servers, no GPU. CPU-only Python checks were allowed. Repo: vLLM-Omni tag v0.34.0, HEAD `06a28821a`.

Four sub-investigations were run. Each has its own log in this directory:

| Sub-report | Covers |
|---|---|
| [2026-11-26_gate-reason-and-text-warmup-hidden-states.md](2026-11-26_gate-reason-and-text-warmup-hidden-states.md) | why the audio-output gate exists; test coverage; stage routing for text-only; hidden-state save path; verdict on text-only warmup then audio query |
| [2026-11-26_tmrope-image-items-prefix-invariance.md](2026-11-26_tmrope-image-items-prefix-invariance.md) | image-item mrope positions; prefix invariance; `use_audio_in_video` on images; token counts (CPU check) |
| [2026-11-26_encoder-cache-capacity-streaming-frames.md](2026-11-26_encoder-cache-capacity-streaming-frames.md) | encoder cache size per YAML; frames that fit; eviction; warmup abort; KV capacity estimate |
| [2026-11-26_measurement-hooks-process-cache-timing.md](2026-11-26_measurement-hooks-process-cache-timing.md) | process model and sitecustomize; `num_cached_tokens`; abort semantics for warmup blocks; encoder hook; mm-processor-stats |

Marks: **VERIFIED** = code read, or a CPU run, or a repo command output shown here. **INFERRED** = follows from code, not run. **UNVERIFIED** = needs a live engine or GPU. Items marked "(me)" were re-checked by the lead agent in this session; the rest rest on the sub-report's file:line citations.

---

## Summary of answers

1. **The audio-output gate has no stated technical reason.** It was added in commit `eb1699bfc` (PR #5894). The PR scopes it as: "Audio-output sessions on this endpoint stay on the legacy path and do not start warmup." (me: the gate is at `video_stream_base.py:298` in that commit.) VERIFIED for the code; the PR text comes from the sub-agent's read of the GitHub API.
2. **Nothing in the tests covers the case we care about.** Two e2e tests combine audio output with a stage-0 prefix hit, but they replay an identical prompt. None covers a partial-prefix hit with audio, or a text-only warmup followed by an audio query. Neither has been run.
3. **A text-only warmup followed by an audio query is likely to work at the engine level.** It is unverified on GPU. The blocker is in the handler: warmups pass the session's `modalities` (`video_stream_base.py:1392, 1412`). Changing that to `["text"]` is the minimal change (see section 3).
4. **Image frames are position-stable.** Appending a frame does not change the positions of earlier tokens. This is verified on three implementations (vLLM-Omni, upstream vLLM 0.34, HF reference) with a CPU check. Positions of text after the last frame do shift.
5. **Token counts are corrected.** 640x480 gives 300 tokens. 1280x720 is resized to 1280x704 and gives 880 tokens, not the 900 estimated earlier.
6. **The encoder cache is large.** Its size is 62720 embeddings for all four stage-0 YAMLs. That fits 209 frames at 640x480 and 71 at 1280x720. The 50-frame default buffer fits at both. This is a CPU-derived figure and must be confirmed in a live log.
7. **Warmups are serialized, not one per frame.** A new warmup starts only after the previous one finishes (me: `video_stream_base.py:474-475`, `:515-517`).
8. **Measurement hooks are identified.** The per-request encoder hook is `ModelState.execute_mm_encoder` (me: `vllm/v1/worker/gpu/model_states/interface.py:191`), which is keyed by engine request id. `num_cached_tokens` is populated for stage 0 under MRv2, but the warmup path discards it. Aborted warmups keep their full blocks in the prefix cache.
9. **`--enable-mm-processor-stats` is not a `vllm serve` flag** in vLLM 0.34 (me: the generated parser has no such option). It can only be set through a stage YAML's engine_args, which is unverified end-to-end.

---

## 1. Audio gate: reason, tests, routing, hidden states

### 1.1 Reason for the gate
- Introduced in `eb1699bfc` "[FEAT][Fronted] Support incremental prefill for realtime video streaming (#5894)". The commit body contains only sign-off lines. (me: `git show --stat eb1699bfc` lists `video_stream_base.py` and the docs; line 298 contains the gate.)
- The PR body defines scope only. No technical reason is given (sub-report, from the public PR body).
- `docs/serving/video_stream_api.md:117-118` repeats the scope statement.
- **Conclusion:** the gate is a scope decision. Nothing in code, docs or PR states a technical blocker.
- Note: `git log -S` on the gate string shows `06a28821a` (the v0.34.0 backport), because that backport added the whole file as one commit (1514 insertions). The gate's real origin is `eb1699bfc`.

### 1.2 Tests
- `tests/e2e/online_serving/test_qwen3_omni.py:263` `test_thinker_prefix_caching_audio_output`: checks a prefix hit with audio output, not the audio itself.
- `:372` `test_thinker_prefix_caching_async_chunk_identical`: checks the hit and the audio bytes; the prompt is identical on both requests.
- Not covered: partial-prefix hit with audio, or text-only warmup followed by audio query.
- Stale comment: `test_qwen3_omni.py:33-38` says the production YAML leaves the thinker prefix cache off under MRv2. The YAML enables it for CUDA (`qwen3_omni_moe.yaml:40`), so the comment is out of date.
- No test was run.

### 1.3 Stage routing for text-only requests
- Final stage for `["text"]` is 0 (`entrypoints/utils.py:181-216`).
- Stage 1 (talker) is not prewarmed (`orchestrator.py:3680`) and not forwarded (`orchestrator.py:2532`, condition `stage_id < final_stage_id`).
- Stage 0 may still publish a talker-bound payload on the MRv2 native data plane. Nothing consumes it. INFERRED.

### 1.4 Hidden-state saving
- The save is not gated per request. The check is only on cache presence, pooling, and PP rank (`core/prefix_cache/runner_mixin.py:168-222`, `worker_v2/prefix_cache.py:167-214`).
- The thinker opts in at `model_executor/models/qwen3_omni/qwen3_omni.py:297`. Default `needs_full_hidden_states=True` (`core/prefix_cache/interface.py:145`).
- Default runner is MRv2. V1 is refused (`engine/arg_utils.py:484-485`).
- The block hash has no modality term (`v1/core/kv_cache_utils.py:633-660`). On a hit, the rewrite restores full-prompt hidden rows into the thinker payload (`worker_v2/prefix_cache.py:60-88`). The talker keeps no cache of its own.

### 1.5 Verdict for text-only warmup, then audio query
- **Likely works at engine level** (INFERRED). Conditions: identical token prefix and mm hashes; no preemption (a preempted hit loses hidden states, per `video_stream_api.md:147-150`); the blocks are still resident.
- **Blocker is in the handler.** Change warmup modalities to `["text"]` in `_drive_warmup_generate` (`video_stream_base.py:1359`) and in `_prefill_context` (`:1392, :1412`), and relax the gate at `:298`.
- **Failure mode to guard against:** the audio can degrade silently while text looks normal. A check must compare audio and codes, not just cache counts.

### 1.6 GPU test to run (not done; needs approval)
1. Text-only warmup (`max_tokens=1`), then an audio-output query whose prefix hit is partial (cached > 0, block-aligned, less than the full prompt).
2. Compare the talker conditioning and output codes with a cold run of the same prompt. The talker seed is 42 (per the sub-report).
3. Confirm the hidden-state rewrite: `full_payload_prefix_cache_keys` contains `"hidden"`.
4. Record `num_cached_tokens` for the query.

---

## 2. Image-item mrope positions and prefix invariance (CPU-verified)

- **Prefix invariance: yes.** For prompts A = [system, history, img1], B = [..., img1, img2], C, the positions of tokens in A's prefix are identical in B and C. Checked against vLLM-Omni's implementation, upstream vLLM 0.34's, and HF `get_rope_index`. Positions are computed left to right with no look-ahead (`qwen3_omni_moe_thinker.py:1822-1913`). VERIFIED (CPU run in `/tmp/audit_tmrope/check_positions.py`, log `run1.log`).
- **Pixels are independent across images in a batch.** Max difference 0.0. VERIFIED.
- **What does shift:** text after the last frame, including the question. Example: adding one 1280x720 frame moves the question's position from 72 to 114. This matters only if a suffix is cached, which the design does not do. VERIFIED.
- **Image t-index:** each image has `grid_t = 1` (`image_processing_qwen2_vl.py:258`). The single frame is duplicated only inside the patch vector (`:203`). All image tokens share one t value, the running position at image start. Text after an image advances by `max(H_merged, W_merged)` plus the vision-end token, not by token count. VERIFIED.
- **Video comparison:** video t steps by `second_per_grid_ts * 13` per temporal patch. Text after a video depends on the video's length. Per appended frame the position budget is 22 slots at 640x480 and 42 at 1280x720.
- **`use_audio_in_video` on images:** changes only the mm hash. Identical `input_ids` and `pixel_values` with and without the flag. vLLM-Omni's thinker reads it only for video items. VERIFIED.
  - Under incremental prefill the flag is set on every warmup and query, so the hash is stable.
  - In legacy mode it toggles with audio, which would make image prefix-cache entries miss. INFERRED, not run.
- **Token counts:** 640x480 gives grid `[1, 30, 40]` and 300 tokens. 1280x720 gives `720/32 = 22.5`, which rounds to 22, so h = 704, grid `[1, 44, 80]`, and 880 tokens. The earlier 900 was wrong. VERIFIED.
- **Not verified:** the KV block-hash effect of positions; live WS frame sizes; real audio-in-video pairing; processor `second_per_grid` values for video.

---

## 3. Encoder cache and KV capacity (CPU-derived; not live)

### 3.1 Size
- Stage-0 encoder cache = `max(max_num_batched_tokens, 62720)` embeddings for all four YAMLs (`qwen3_omni_moe.yaml`, `qwen3_omni_moe_thinking.yaml`, `qwen3_omni_moe_mrv2.yaml`, `qwen3_omni_duplex_mrv2.yaml`).
- `max_num_batched_tokens` is 32768 where set (default, MRv2, duplex). The thinking YAML omits it and the sub-report reports a hardware default of 16384/8192. The sub-report notes that the YAML comment about a "32768 default" is wrong.
- The 62720 is the video maximum: Qwen3-Omni processing info is frame-based at `max_model_len` 65536, giving 10 frames at 3584x3584 (5 x 12544 embeddings per frame group). The image maximum is 12544, so images do not set the size.
- Derived with a CPU engine-config run and `MultiModalBudget`. A live stage log should print "Encoder cache will be initialized with a budget of 62720 tokens". UNVERIFIED until that line is seen.

### 3.2 Frames that fit
| Resolution | Tokens per frame | Frames that fit in 62720 |
|---|---|---|
| 640x480 | 300 | 209 |
| 1280x704 (from 1280x720) | 880 | 71 |

A 50-frame buffer uses 15000 (640x480) and 44000 (720p) embeddings. Both fit.

### 3.3 Eviction and re-encoding
- (a) A cached frame is not re-encoded. `check_and_update_cache` returns True and the worker skips it.
- (b) Victims are unreferenced entries, oldest-released first (LRU over unreferenced). A buffered frame can be evicted and re-encoded only under pressure. At 62720, stale frames are evicted first, so steady state is about one new encode per warmup.
- Sensitivity: at 32768 with 720p frames, about 22 re-encodes per warmup. (Sub-report arithmetic.)
- (c) Aborting a warmup calls `free()` through `finish_requests`. Its references are released, and the entries stay cached as freeable until evicted.
- Chunked prefill: a re-check for the same request returns True without double-counting. Entries still referenced are not evictable.
- Warmups are serialized (see summary item 7).

### 3.4 KV capacity (order of magnitude)
- Thinker KV: 96 KiB per token (sub-report).
- H100 80 GB: roughly 87k–109k tokens, order 1e5. H200: about 0.7M. Derived; memory overhead UNVERIFIED.
- Implication: 50 frames at 880 tokens is about 44k tokens of KV before history. Two concurrent sessions at 720p could evict each other's prefix blocks on an 80 GB card. INFERRED from the derivation above.

---

## 4. Measurement hooks

### 4.1 Process model
- API process (orchestrator and WebSocket handler), then one spawned `StageEngineCoreProc` per stage. Spawn is forced (`engine/stage_init_utils.py:783-792`). The environment is inherited (`multiprocessing/util.py:450-462`).
- At TP=1 the scheduler, `EncoderCacheManager`, the KV pool and the MRv2 model runner (which runs the vision encoder) share one stage process (`UniProcExecutor`). TP>1 adds spawned workers.
- A `PYTHONPATH` `sitecustomize.py` loads in the API process and in every stage process. Confirmed by a scratch spawn test in `/tmp/audit_meas`. It does not load under `-I` or `-S`. No engine log in the repo proves this yet. VERIFIED for the spawn path; UNVERIFIED in a live engine.

### 4.2 `num_cached_tokens`
- Set at first scheduling (`v1/core/sched/scheduler.py:1291`) and emitted with the first output. The MRv2 runner is the only omni runner (V1 is refused), and it populates the field for `max_tokens=1` requests too.
- Gaps:
  - Async-chunk non-final stage 0 can drop the stats on non-stopping steps (`omni_ar_scheduler.py:1624` vs `:1776`). A missing value reads as 0, which looks like a miss.
  - `_drive_warmup_generate` discards all outputs (`video_stream_base.py:1364`). Warmups cancelled before their first token report nothing.
- Reliable source: `KVCacheManager.get_computed_blocks` at admission. Recommended over the output field.

### 4.3 Abort semantics for warmup blocks
- Full blocks hashed at scheduling stay in the hash map after an abort, freed to the LRU tail, until evicted. The query can reuse them if tokens and mm hashes match from block 0. VERIFIED (sub-report, `kv_cache_utils.py`, `block_pool.py`).
- A partially filled last block is not reusable by default.
- Omni adds no KV release on abort. Its prefix cache forces pending writes to land, for Qwen3-Omni under MRv2 only.

### 4.4 Encoder timing hook
- **Recommended:** wrap `vllm.v1.worker.gpu.model_states.interface.ModelState.execute_mm_encoder` (me: method at `interface.py:191`, takes `scheduled_encoder_inputs: dict[str, list[int]]`). It runs per step in the worker, is keyed by engine request id, and happens after scheduler filtering.
- Not suitable: `EncoderRunner.execute_mm_encoder` (me: `encoder_runner.py:150`, takes only `mm_kwargs`, no request ids). The 11-25 audit recommended it; that was incomplete.
- `check_and_update_cache` is a cache decision, not an encoder run. Use it only for hit/miss.
- Warmup attribution: engine request ids for warmups have the prefix `video-warmup-` (`video_stream_base.py:484`), so they can be told apart from queries (`video-`).

### 4.5 `--enable-mm-processor-stats`
- Not a generated CLI option in vLLM 0.34 (me: `EngineArgs.add_cli_args` yields no option matching it). The field exists in `EngineArgs` (`engine/arg_utils.py:771`) and `ObservabilityConfig` (`config/observability.py:96`).
- Reachable through a stage YAML's `engine_args` (sub-report: filter at `stage_init_utils.py:1596`). UNVERIFIED end-to-end.
- Read only through `collective_rpc("get_encoder_timing_stats")`. No HTTP route exposes it. Keys are engine request ids, so warmups can be separated by prefix (INFERRED).

---

## Corrections to the 2026-11-26 audit

| # | Earlier statement | Correction |
|---|---|---|
| C1 | "A new warmup is started on each new frame" | Warmups are serialized and coalesced: a new one starts only after the previous finishes (`video_stream_base.py:474-475, 515-517`). |
| C2 | Prefix invariance was inferred | Verified on CPU across three implementations (section 2). |
| C3 | 1280x720 frame ≈ 900 tokens | 880 tokens (resize to 704). |
| C4 | Encoder cache for the batch path holds "about one video" with the thinking YAML (8192) | All four YAMLs give 62720 embeddings from the video maximum, so about five full-length (11.5k-token) videos fit. This also changes the batch-path analysis in the 11-25 audit; re-derive before relying on it. |
| C5 | `--enable-mm-processor-stats` as a `vllm serve` flag | Not a generated CLI option. Only via stage YAML engine_args (unverified). |
| C6 | Encoder hook `EncoderRunner.execute_mm_encoder` | Use `ModelState.execute_mm_encoder` (has request ids). |
| C7 | V1 `gpu_model_runner.py` referenced in the measurement brief | Does not exist in vLLM 0.34 (`v1/worker/` has `gpu/`, `gpu_worker.py`, `mm_encoder_model_runner.py`). MRv2 is the omni runner. |
| C8 | Gate reason unknown | Scope decision in PR #5894; no technical reason stated. |

The 11-26 streaming audit file has a correction note added at its top.

---

## Implications for the study

1. **Frame-level encoder reuse is testable on the streaming path.** Each frame is its own mm item with its own key (uuid = md5 of bytes). Under the 62720 budget, a 50-frame buffer stays fully cached at both resolutions, so the encoder cache is not the limit at this size.
2. **The likely limit is KV, not the encoder cache.** At 720p, a long buffer plus history on one 80 GB card may evict prefix blocks. This is derived; the 0.7M-token H200 figure is not checked.
3. **Measure the hit with `get_computed_blocks`, not with the output field.** Record `num_cached_tokens` only where the MRv2 path populates it.
4. **Keep `use_audio_in_video` and the frame set fixed during a run.** Both are hash factors; changing either silently turns hits into misses.
5. **Option A (text-only warmups) is the cheapest route to audio-output incremental prefill.** It needs a GPU check (section 1.6) that compares audio and codes against a cold run. Passing only cache counts would miss silent degradation.
6. **Instrumentation to add before any run:** warmup start/finish and cancellation with request ids (`video-warmup-*`); `get_computed_blocks` at admission; a `ModelState.execute_mm_encoder` wrapper that records request id, item count and time.

---

## Still unverified (needs live engine or GPU)

- Text-only warmup producing a usable audio-query prefix hit, and audio equality against a cold run (section 1.6).
- The 62720 budget (the stage log line; section 3.1).
- Whether `sitecustomize` loads in the live stage processes.
- `num_cached_tokens` values on the streaming path.
- KV capacity at 80 GB and 720p.
- Whether `--enable-mm-processor-stats` works through a stage YAML.
- Live TMRoPE for image items versus HF (CPU check only).

**A GPU validation run needs your approval.** Memory notes say the 5880 cards are shared with others, so the run should use a card you confirm is free, and any server started should be stopped afterwards.
