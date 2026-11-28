# M* Qwen3-Omni: incremental audio/video prefill and exactness

## Question asked

Could M* Qwen3-Omni accept audio and video in time chunks, run the audio and vision encoders per chunk on arrival, and prefill the Thinker incrementally, so that at query time only question tokens and decode remain? Would that incremental prefill be EXACT (same tokens and same position ids as a one-shot prefill of the same content) or approximate?

- Q1. TMRoPE position ids: how video, audio and text spans get positions, and whether earlier positions stay fixed when later chunks are appended.
- Q2. Audio/video interleaving (`use_audio_in_video`), what M* would need, and whether an interleaved incremental prefill is exact.
- Q3. How a request maps to walks and graph nodes, whether a walk can be fed over time, and the scheduler condition for a walk to become runnable.
- Q4. Smallest change, size estimate, and what is missing for exactness.

Marks: **VERIFIED** = read in code or reproduced on CPU. **INFERRED** = follows from code read, not executed. **UNVERIFIED** = not checked.

## Scope

- M* at commit `2c8ffde8f` (`git rev-parse --short HEAD`), read only: `/home/chaofan/workspaces/mm_reuse_study/external/mstar` (package `mstar/`). Nothing under `external/` was modified.
- Files read: `mstar/model/qwen3_omni/{qwen3_omni_model.py, submodules.py, config.py, components/rope.py}`, `mstar/model/components/{mrope.py, aut_encoder.py, qwen3_vit.py}`, `mstar/graph/base.py`, `mstar/model/base.py`, `mstar/api_server/{data_worker.py, entrypoint.py, openai/adapters.py}`, `mstar/model/media_limits.py`, `mstar/model/qwen3_omni/video_reader.py`.
- Reference code (mstar env, read only): `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py` (transformers 5.24.1), `qwen_omni_utils/v2_5/vision_process.py` (qwen_omni_utils 0.0.9).
- Checkpoint: `config.json` only, from `models/cache/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/*/`. No weights loaded.
- Background used: `agents_results/2026-11-25_qwen3-omni-g0-code-audit.md`, `agents_results/2026-11-26_tmrope-image-items-prefix-invariance.md`.
- CPU check: `/tmp/audit_mstar_q3/check_positions.py`, log `/tmp/audit_mstar_q3/run1.log`. Command: `env -u MODEL_SEQ_LEN CUDA_VISIBLE_DEVICES= /home/chaofan/workspaces/mm_reuse_study/envs/mstar/bin/python -B /tmp/audit_mstar_q3/check_positions.py`. Exit 0.

## Findings

### Q1. Position ids (TMRoPE)

**Where they are computed (VERIFIED)**

- Helpers: `components/rope.py:166-184` (`get_rope_index_text`, sequential on all three axes) and `components/rope.py:196-259` (`get_rope_index_vision`). For a video, the temporal axis is `start + t * seconds_per_grid * position_id_per_seconds` (`rope.py:247-254`). For an image it is constant `start`.
- Thinker prefill (`submodules.py`):
  - The stream counter is read at `:407`: `start_pos = position(self.kv, rid, "main")`.
  - Audio row (`_audio_row`, `:492-525`): all positions are `arange(n) + start_pos`, and the counter advances by `n`.
  - Vision row (`prefill_vision`, `:412-455`): BOS at `media_pos = start_pos + len(before)`; vision positions from `media_pos + 1` (`:430`); EOS at `max + 1` (`:445`); text after the media from the spliced row (`_spliced_row`, `:527-566`).
- Layout counter (`qwen3_omni_model.py:1175-1186`): a video span's advance is `max(positions) + 2 - position`, using `get_rope_index_vision(grid, position + 1, pps, seconds_per_grid=per_grid)`.
- Text rows use the planned counter (`components/mrope.py:20-44`; `submodules.py:726-727`).
- HF reference: `get_rope_index` at `modeling_qwen3_omni_moe.py:220-445`. Text after a video starts at `max + 2` (EOS at `max + 1`, then `st_idx = max + 1` at `:442-446`). M*'s `end_pos_base` matches this.

**What a video span's positions depend on (VERIFIED from code, and from the CPU check)**

Let `origin` = counter at the video's BOS, plus 1. Let `spg` = seconds per temporal group, and `h', w'` = merged grid.

- Temporal index of group `t`: `origin + t * spg * 13`. `origin` is the stream counter at the start of the video, not 0 and not the total length. `13` is `position_id_per_seconds` from the checkpoint (`thinker_config`, value 13; `config.py:128` default of 25 is overridden by the loader, CPU check confirms `ThinkerConfig.from_dict` gives 13).
- `spg = temporal_patch_size / sample_fps = 2 / sample_fps` (`qwen3_omni_model.py:1110-1122`). `sample_fps = nframes / total_frames * video_fps` (`qwen_omni_utils/v2_5/vision_process.py:216/321/367`). `nframes = total/video_fps * fps`, clamped to [4, min(768, total)] and floored to even (`vision_process.py:144-182`). So **spg depends on the total length of the video**.
- Spatial indices: `origin + row/col` on the merged grid. The grid depends on the per-frame resize, and `max_pixels = max(min(602112, 90316800 / nframes * 2), 105369)` (`vision_process.py:450-451`). So **for more than 300 sampled frames (about 150 s at 2 fps), the frame size shrinks with the total frame count**.
- Which frames are sampled: `linspace(start, end, nframes)` over the whole file (`vision_process.py:317`). **Depends on total length.**

**Audio span (VERIFIED)**: sequential counter, depends only on its own token count. 13 tokens per 100 mel frames (`aut_encoder.py:70-75`), matching HF `_get_feat_extract_output_lengths` (`modeling_qwen3_omni_moe.py:150-160`) on all tested lengths.

**Text after a video (VERIFIED, CPU)**: `after = origin + max((T-1) * spg * 13, h'-1, w'-1) + 2`, where `T` is the number of temporal groups. So the first text after a video depends on the video length. CPU: one-shot `[text 30, video (8,44,80), text 12]` gives question position 124 = 31 + 91 + 2.

**Prefix question (VERIFIED, CPU, `check_positions.py` Q1a-Q1c)**

- The M* emulation (`mstar_positions`, built from `submodules.py` and `rope.py`) matches HF's own `get_rope_index` on the one-shot case: length 7084, max abs diff 0.0.
- **Positions of earlier tokens do not change when later temporal groups are appended inside the same video attachment**, given fixed `origin` and `spg`. CPU: the first 3551 positions of the `T=8` one-shot equal those of the `T=4` prefix. The temporal index does not depend on `T`; the question's position does.
- **But the real data path does not keep `spg` fixed.** CPU: a 10.0 s clip at 30 fps gives `spg = 1.0000`; a 9.9 s clip gives `spg = 1.1000`. The same first group therefore gets different temporal positions. A 400 s clip gives `spg = 1.0417`. So the earlier positions are NOT prefix-stable in the real pipeline.
- **Appending a chunk as its own attachment (M*'s current walk scheme) changes later positions and tokens (VERIFIED, CPU Q1c)**:
  - Chunk 1 positions are identical.
  - Chunk 2's first temporal index is 73 (appended) vs 83 (one-shot): each attachment's origin is the counter, not the video start.
  - The question is at 114 vs 124.
  - Token counts are 7086 vs 7084, because each attachment adds its own BOS and EOS.

**Verdict Q1**: the TMRoPE formula is prefix-stable for appended temporal groups (YES, VERIFIED). The M* data path is not (NO, because `spg`, the per-frame resize, and the frame indices depend on the total length; VERIFIED). Per-attachment chunks are not exact (NO, VERIFIED).

### Q2. Audio/video interleaving

**M* today (VERIFIED)**

- `load_video` starts with `# TODO: support audio in video` (`qwen3_omni_model.py:919`). `process_prompt` extracts no audio from the video (`:1107-1124`).
- The API takes whole files per part: `image_url`, `video_url`, `audio_url`, `input_audio` (`openai/adapters.py:144-154`). There are no chunk parts.
- `seconds_per_chunk: 2` is in the checkpoint `thinker_config`, and no M* code reads it (grep finds nothing).

**HF reference (VERIFIED)**

- With `use_audio_in_video`, video groups are at `st + t * spg * 13` (`modeling_qwen3_omni_moe.py:~387, 410`). Audio tokens are at `st + i` (`:~400`, `arange(audio_len) + st_idx`).
- The two are merged by the temporal row, with video winning ties (`:420`).
- Audio runs at 13 tokens per second, matching `position_id_per_seconds = 13`. So audio token `i` lands on time `i/13` s.
- CPU: for a 2 s clip with grid `(2,44,80)` and 200 mel frames, audio tokens take temporal positions 7..20, and audio token 13 shares position 20 with video group 1. Merge order (CPU, from `:420`): `V(t0)` 880 tokens, `A0..A12`, `V(t1)` 880 tokens, `A13..A25`.

**What M* would need (INFERRED from the code read)**

1. Chunk intake: today's intake takes whole files only (see `adapters.py:144-154`).
2. Audio extraction from the video track. M*'s `load_video` does not do this. The library is unchecked.
3. A 2 s chunker. Per chunk: video groups at `origin + t*spg*13`, and audio at `origin + 13*time`. This needs explicit positions. `ARNodeInputs.custom_pos_ids` supports that (`mrope.py:37-43`). `_audio_row` (`submodules.py:492-525`) uses the sequential counter instead, so it would need changing.
4. A layout with `<vision_start><audio_start> ... <audio_end><vision_end>` and the interleaved ordering, in `_state_layout` (`qwen3_omni_model.py:1175-1201`) and `_PLACEHOLDER_TOKENS`.
5. Keyed spans updated (`qwen3_omni_model.py:1199-1201`).

**Is an interleaved incremental prefill exact? (positions: yes if `origin` and `spg` are known; encoder: no for 2 s chunks)**

- Audio attention is bidirectional within 8 s windows of 104 tokens (`aut_encoder.py:1-20`, `window_tokens` at `:80-82`, `window_lengths` at `:84-88`, per-window SDPA at `:144-156`). CPU: `window_lengths(260) = [104, 104, 52]`. Tokens 0..25 (the first 2 s) attend to tokens 26..103 (the next 6 s). So a 2 s chunk's audio encoder output depends on future audio. **Exact only at 8 s boundaries aligned to the clip start.** The conv stage is local per 100-frame chunk (`aut_encoder.py` docstring), so 1 s granularity is safe for conv (INFERRED).
- Mel-spectrogram edge effects at chunk boundaries: UNVERIFIED.

### Q3. Node structure and scheduling

**Request to walks (VERIFIED)**

- One Thinker step per attachment, with the text before it: `_thinker_steps` (`qwen3_omni_model.py:208-221`).
- `_MM_WALKS` (`:872-883`): `audio` to `prefill_audio`; `image` and `video` to `prefill_vision`.
- `prefill_audio = Sequential([audio encoder, Thinker])` (`:556`). `prefill_vision = Sequential([vision_encoder, Thinker])` (`:588-622`).
- The Thinker `GraphNode` input names are `vision_embeds, deepstack, video_second_per_grid, image_grid_thw, text_inputs` (`:600-605`) and `audio_embeds, text_inputs` (`:~560-575`).
- Chunking (`get_chunking_policies`, `:480-493`; `MAX_BATCH_TOKENS = 1024`, `submodules.py:566`). `split_inputs` (`submodules.py:568`) cuts a row that was prepared whole. It does not grow it.

**Feeding over time (VERIFIED: no existing mechanism for prefill)**

- Every walk's inputs are produced at intake for the whole request: `process_prompt` over the complete request at `data_worker.py:1012`, and KV layouts from the full prompt at `data_worker.py:1030-1039`.
- Client streams exist (`entrypoint.py:1029` `stream_input`; `consumes_stream` nodes in `qwen3_tts`, `nemotron_duplex`, `waypoint`, e.g. `qwen3_tts_model.py:626-650` `text_in` per talker loop iteration). They feed decode loops, not prefill walks.
- Qwen3-Omni declares no client connections: `get_partition_topology` lists only Thinker to Talker to Code2Wav (`qwen3_omni_model.py:780-806`).

**Scheduler condition for a walk to become runnable (VERIFIED)**

- `ReadySignals.update` sets `is_ready` only when `input_names ⊆ ready_names` (`graph/base.py:176-192`).
- The worker registry adds a node to `ready_names` only when `node.ready_signals.is_ready` (`graph/base.py:835-848`, `register_ingested_input`). It is queued as streaming-ready only when the node has streamed inputs.
- Streamed inputs come from client streams (`model/base.py:443`, `_register_streaming`). Qwen3-Omni has none, so **a walk runs only when all its declared input tensors for the request are present**.

**Verdict Q3**: no existing mechanism extends a prefill walk or its KV over time. The nearest existing mechanism is client streams with `consumes_stream`, which is built for decode loops. Whether the engine can grow a KV layout mid-request is UNVERIFIED (layouts are computed at intake, `data_worker.py:1030-1039`).

### Q4. Smallest change and exactness

**Changes for true streaming (sizes are rough INFERRED estimates; engine parts UNVERIFIED)**

| Part | Where | Estimate |
|---|---|---|
| (a) Accept chunked input for one session | `openai/adapters.py`, `entrypoint.py` (session plumbing exists at `:1029-1090`), `qwen3_omni_model.py:780-806` (add a CLIENT connection), `consumes_stream` | ~80-150 lines |
| (b) Per-chunk vision and audio encoders | a chunk processor (fixed-rate sampling, `video_processor`, `feature_extractor`); existing nodes `qwen3_omni_model.py:556-580` and `:588-622` run per chunk | ~150-250 lines |
| (c) Keep Thinker KV and extend it | streamed prefill in `submodules.py:365-460` and `:568-600`; incremental `_state_layout` (`qwen3_omni_model.py:1175-1201`); layout growth in the engine (UNVERIFIED) | ~200-400 lines plus engine work |
| (d) Query time: question tokens and decode only | existing `prefill_text` (`submodules.py:~380-405`) with the planned counter | ~50 lines |

Total roughly 500-900 lines, and the engine KV-growth part is the unknown.

**What is missing for exactness**

1. **Sampling depends on total length.** `nframes`, `linspace` frame indices, `spg`, and the per-frame resize (above 150 s) all depend on the whole file (VERIFIED). A live stream cannot reproduce today's one-shot tokens. Exactness needs either the total length known up front (`probe_video`, `media_limits.py:522`, already probed in `load_video` at `qwen3_omni_model.py:921-927`) or a new length-independent rule (fixed fps, fixed resolution, fixed origin). The second changes outputs relative to today's M*.
2. **Sentinels and origin per attachment.** One BOS and one EOS for the whole video, with explicit per-chunk positions. Per-attachment chunks are not exact (VERIFIED).
3. **Audio 8 s windows.** 2 s audio chunks are not exact. Exact only at 8 s boundaries aligned to the clip start (VERIFIED from code, not run).
4. **Audio-in-video** is unimplemented, and its ordering must match HF's merge (CPU-verified reference).

**Smallest EXACT option (recommended first test)**

Keep the one-shot layout, but make the video prefix reusable. Prefill `[system, text, video]` at upload, keyed by the whole-file digest, and at query time prefill only the question and decode.

- Extend `keyed_spans` beyond the video. Today it stops at the leading text (`qwen3_omni_model.py:1199-1201`, with the comment "keying past it has never been run or tested"). Video keying is INFERRED, not checked.
- Causal attention plus the position math above (the question's positions come from the full layout at query time) make this exact in tokens and positions. Numerics match up to kernel batch-shape effects (INFERRED).
- Size: about 60-150 lines in `qwen3_omni_model.py`. Also needed: a prefix-only request path (UNVERIFIED in the engine), and the rebuild policy (`qwen3_omni_model.py:495-507`; a video prompt with prefix cache on is not rebuildable, `:496-499`). Skipping the vision encoder on a cache hit is an engine/graph change (UNVERIFIED size, since `prefill_vision` always runs its encoder node, `:588-622`).
- Multi-rank caveats from the background (`serving.mdx:186, 492`) apply.

**Exactness verdict**: streamed incremental prefill of a growing video is NOT exact against today's one-shot M*. Exact variants are (i) whole-file prefix caching at upload, which saves question-time work but not encoder work unless the encoder is also skipped; or (ii) a new length-independent sampling rule, which changes outputs.

## CPU check

- Script: `/tmp/audit_mstar_q3/check_positions.py`. Log: `/tmp/audit_mstar_q3/run1.log`. Exit 0.
- The HF reference is the unbound `get_rope_index` on a stub object with the checkpoint's token ids, `spatial_merge_size=2`, and `position_id_per_seconds=13`.
- Key outputs:
  - Q1a: M* emulation vs HF, length 7084, max abs diff 0.0. Question at 124.0 in both.
  - Q1b: first 3551 positions equal for the `T=4` prefix vs one-shot `T=8`: True.
  - Q1c: appended attachment: chunk 2 first temporal 73.0 vs 83.0 one-shot. Question 114 vs 124. Token counts 7086 vs 7084.
  - Q1d: nframes and spg. 10.0 s: 20, 1.0000. 9.9 s: 18, 1.1000. 1.5 s: 4, 0.7500. 400 s: 768, 1.0417.
  - Q1e: per-frame resize for 720x1280. nframes 100 and 300: `max_pixels` 602112, 560x1008. nframes 400: 451584, 504x896. nframes 768: 235200, 336x644.
  - Q2a: `tokens_for_frames` matches HF on all tested lengths (no mismatches). `window_lengths(260) = [104,104,52]`.
  - Q2b: audio temporal positions 7..20; audio token 13 shares position 20 with video group 1.

## Could not verify

- Engine KV-layout growth mid-request, and whether a prefix-only request (no question) can be issued through the API (UNVERIFIED).
- Whether video keys can be enabled past the video span (`keyed_spans`), and the prefix-cache hit path for video (INFERRED).
- Numeric equality of audio encoder outputs for 2 s chunks vs one-shot (needs weights; only the window structure was verified).
- Mel-spectrogram edge effects at chunk boundaries (UNVERIFIED).
- Whether the vision encoder output for a temporal group is independent of other groups numerically (structure verified at `qwen3_vit.py:54-57`, `submodules.py:224`; not run).
- Whether the second video resize (`process_prompt`, `qwen3_omni_model.py:1120-1124`) depends on `nframes` (INFERRED no).
- Whether M*'s PyAV fallback (`video_reader.py`) samples the same frames as decord/torchcodec (UNVERIFIED).
- Whether a deployment sets `MODEL_SEQ_LEN` (M* does not set it; the default 128000 was assumed in the CPU check).
- Image positions and the vLLM-Omni side are out of scope.
