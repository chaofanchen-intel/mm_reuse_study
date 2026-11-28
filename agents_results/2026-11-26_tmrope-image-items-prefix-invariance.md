# TMRoPE for image items: prefix invariance, t-index, use_audio_in_video (Qwen3-Omni)

## Question(s) asked

- Q1. Where are Qwen3-Omni thinker mrope positions computed; how are image items handled vs video items; how image t-index is assigned and how text positions advance after an image.
- Q2. Prefix invariance: do earlier tokens' positions depend on later tokens? For prompt A = [system, history, img1] vs B = [system, history, img1, img2] (+ question), are A's positions identical in B? Do image pixel tokens depend on other images? Does the per-image token count depend on position? Confirm with a CPU check at 640x480 and 1280x720.
- Q3. What does `use_audio_in_video` change (processor kwarg, mm hash factors, positions, placeholders)? Does it change image-item tokens or positions, or only the hash? Does vLLM-Omni's thinker read it for image items?
- Q4. Image temporal index: t=0 or t=i? Single frame duplicated to temporal_patch_size=2? Video t-step (second_per_grid_ts, position_id_per_seconds=13), and does the text after a video depend on its length? Implication for "a frame appended later shifts later text but not earlier ones".
- Q5. Per-image vision token count formula from preprocessor_config.json (patch 16, merge 2, min_pixels 3136, max_pixels 12845056), verified on CPU for 640x480 and 1280x720.

## Scope

- vLLM-Omni v0.34.0 at `/home/chaofan/workspaces/mm_reuse_study/external/vllm-omni` (read only; path root `vllm_omni/`).
- vLLM installed at `/home/chaofan/workspaces/mm_reuse_study/envs/vllm-omni/lib/python3.12/site-packages/vllm` (read only).
- transformers 5.22.2 in the same env (read only).
- Checkpoint files read: `config.json` and `preprocessor_config.json` only, plus the tokenizer/feature-extractor files that `AutoProcessor` needs (no weights).
- No GPU, no server, `CUDA_VISIBLE_DEVICES=` for all Python, `python -B` (no .pyc written into the repo or vLLM install).
- Scratch: `/tmp/audit_tmrope/check_positions.py`, log `/tmp/audit_tmrope/run1.log`.
- The scratch script file was edited on disk during this session. The reported run used the file's final content (the run was done after that edit).

## Findings

### Q1 (mrope code path and image vs video handling)

- VERIFIED. The thinker that vLLM-Omni serves is `Qwen3OmniMoeThinkerForConditionalGeneration` in `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py` (class at L1159). The outer model `qwen3_omni.py:687-697` delegates `get_mrope_input_positions` to `self.thinker` (L697).
- VERIFIED. Per-request call sites pass the full prefill token list and mm features: `vllm/v1/worker/gpu/mm/rope.py:66` and `vllm_omni/worker_v2/staged_writes.py:323`.
- VERIFIED. The vLLM-Omni class overrides the upstream vLLM 0.34 function. Upstream is `vllm/model_executor/models/qwen3_omni_moe_thinker.py:2246` (get_mrope), `:2050` (iter_mm_features). The two implementations differ in form (upstream writes an explicit eos position at L2277/2295 and sets `st = offset + image_len + 1` at L2297; vLLM-Omni sets `st = offset + placeholder_len` at L1869 and has no separate eos). Both give identical positions on every case I ran (see CPU check).
- VERIFIED. `iter_mm_features` (omni L1640-1700). Image branch L1663-1674: `t, h, w = image_grid_thw`; `grid_t = t`; `grid_h = h // spatial_merge_size` (L1670); `grid_w = w // spatial_merge_size`; `t_factor = position_id_per_seconds` (L1672, which is 13 per config). Video branch L1676-1693: `t_factor = second_per_grid_ts * position_id_per_seconds` (about L1692), `second_per_grid_ts` defaults to 2.0 when absent.
- VERIFIED. `get_mrope_input_positions` (omni L1822-1913). Image branch L1857-1869: `np.indices((grid_t, grid_h, grid_w))`; if `t_factor != 1.0` the t-index is multiplied by `t_factor` (t=0 for images, so no effect); positions are `indices + st_idx`; then `st = offset + placeholder_len` (L1869). Text between items gets `arange(text_len) + st_idx` where `st_idx = max(previous positions) + 1` (text branch at top of the loop).
- VERIFIED (config). `thinker_config.position_id_per_seconds = 13`, `spatial_merge_size = 2`, `temporal_patch_size = 2`, `patch_size = 16`, `image_token_id = 151655`, `vision_start = 151652`, `vision_end = 151653`, `video_token_id = 151656`, mrope_section [24,20,20] (config.json, `thinker_config`).
- VERIFIED (HF reference). `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:219` (`get_rope_index`). Image branch: `t_index = arange(grid_t) * 1 * position_id_per_seconds` (L369), so image t = 0 when grid_t = 1. Video branch: `t_index = arange(grid_t) * second_per_grids * position_id_per_seconds` (L385-386). Helper `get_llm_pos_ids_for_vision` at L162. My CPU run matched this HF reference exactly on all cases (see below).

**Answer to the image vs video question.** Image: grid_t = 1, so all tokens of one image share one t-value, and that t-value equals the running position at the image's start (`st_idx`). Relative to the image, t = 0. Video: t-values step by `second_per_grid_ts * 13` per temporal patch (26 for 2.0 s/patch). Text after an image advances the running counter by `max(H_merged, W_merged)` (plus the vision_end token), not by the number of image tokens.

### Q2 (prefix invariance and per-image independence)

- VERIFIED (code). Positions are computed left to right. Each feature's positions depend only on `st_idx` (from earlier items) and its own grid (omni L1822-1913). There is no look-ahead. The only order-dependent pairing is `_get_audio_for_video_mapping` (omni L1613-1638), which pairs video items with audio items; it does nothing for image-only prompts.
- VERIFIED (CPU). A = [sys(20) + hist(30) + VS + img640 + VE + Q(10)], total length 362. B = [sys + hist + VS + img640 + VE + VS + img1280 + VE + Q], total length 1244. C = [sys + hist + VS + img640 + VE], length 352.
  - `A[:, :352] == B[:, :352]`: **True** (all 3 position rows).
  - `A[:, :352] == C`: **True**.
  - HF `get_rope_index` gives the same result for A and B: **True**.
  - Upstream vLLM implementation gives the same result for A and B: **True**.
- VERIFIED (CPU). The question text shifts when a frame is appended. In A the Q starts at position 72; in B it starts at 114.
- VERIFIED (code and CPU). Per-image pixels are independent of the rest of the batch. `image_processing_qwen2_vl.py:212-258` groups images by shape and calls `smart_resize` per image (L63-87). CPU: pixel_values of 640 and 1280 are identical whether processed alone or in a batch of [640, 1280] or [1280, 640] (max abs diff 0.0).
- VERIFIED (CPU). The token count per image depends only on its own size, not its position: 640x480 gives 300 tokens and 1280x720 gives 880 tokens, both as the first image and as the second image in B.

**Answer: appending an image does not change any earlier position.** The stable prefix is everything up to and including the last frame's vision_end token. The question and any other text after the last frame do shift.

### Q3 (use_audio_in_video)

- VERIFIED. HF processor: `processing_qwen3_omni_moe.py:154` pops the flag from `videos_kwargs`. It is used only in the video placeholder expansion (L245-258). The image placeholder path (L242) ignores it.
  - CPU: image-only prompt with `use_audio_in_video=True` vs no flag gives **identical input_ids** (300 `<|image_pad|>` tokens) and **identical pixel_values**.
- VERIFIED. vLLM upstream prompt updates: image replacement is `get_replacement_qwen2_vision(modality="image")` (`qwen3_omni_moe_thinker.py:1486`), with no flag. The flag is read only for the video replacement (L1489; `video_replacement_fn` chosen at about L1516-1523).
- VERIFIED. vLLM upstream `_maybe_apply_prompt_updates` (L1323-1360) sets `use_audio_in_video` only from `mm_kwargs["video"]` items (L1336-1343). Image-only: False.
- VERIFIED. vLLM-Omni thinker: image replacement at omni L965-978 (no flag). `_get_request_video_use_audio_in_video(hf_processor_mm_kwargs, num_videos)` at L974, with `num_videos = len(video_grid_thw) = 0` for image-only input.
- VERIFIED. Positions: the image branch (omni L1857-1869) does not read the flag. The flag is read only for video items (L1676-1693 in `iter_mm_features`, and L1877 in `get_mrope`).
- VERIFIED (vLLM hash). `vllm/multimodal/processing/inputs.py:57-60` copies every non-scoped `hf_processor_mm_kwargs` key (including the flat `use_audio_in_video`) into `mm_processor_kwargs` for the image modality. L70-74 builds `hash_factors`; L91-95 hashes item + factors when factors are present (the client UUID is used directly only when there are no factors, L105).
  - CPU (`MultiModalHasher.hash_kwargs("sha256", image=uuid)` vs with `mm_processor_kwargs={"use_audio_in_video": True}`): **different hashes**, and stable across calls.
- VERIFIED (streaming). `video_stream_base.py:175-181` default `use_audio_in_video = True`. L953-956 (main query): set only if `reuse_active or len(audio_buffer) > 0`. L1397-1400 (warmup): always set when the flag is on. `reuse_active = _incremental_prefill_active(config)` (L928; L297-299: server support and "audio" not in modalities). Under incremental prefill, reuse_active is session-constant, so the flag is set on every query and the image hash is stable.

**Answer: `use_audio_in_video` changes only the hash for image items.** It does not change image tokens, placeholders, or positions. vLLM-Omni's thinker does not read it for image items. It matters only for video items (not used on this streaming path).

- INFERRED. In legacy (non-incremental) mode the main query sets the flag only when that query has audio, so image hashes would differ between audio and no-audio queries, causing prefix-cache misses for image blocks across those queries. I did not run a scheduler or prefix cache to confirm.

### Q4 (temporal index for images vs videos; text after video)

- VERIFIED. Image grid_t = 1. `processed_grids[shape] = [[1, grid_h, grid_w]]` (`image_processing_qwen2_vl.py:258`). The single frame is duplicated only inside the patch vector: pixel_dim = 1536 = 3 channels x 2 temporal x 16 x 16, via `expand(..., temporal_patch_size, ...)` (L203). For 640x480 the pixel rows are 1200 (30x40 patches), not 2400. So the image is NOT a video with two temporal grid steps.
- VERIFIED (CPU). For an image, all image tokens share one t-value. The first image starts at position 51 and its t-row is [51]. The second image in B has t-row [73]. So image t = running position, and relative t = 0.
- VERIFIED (CPU). The image's text advance: 640x480 grid [1,30,40] gives merged H,W = 15,20, so the next text (vision_end) is at 51+20 = 71 and the next text starts at 72. 1280x720 grid [1,44,80] gives merged 22,40, so its vision_end is at 73+40 = 113 and the next text starts at 114.
- VERIFIED (CPU). Video (grid_t = 4, 30x40, 1200 tokens, second_per_grid_ts 2.0): t-values [21, 47, 73, 99], step 26 = 2.0 x 13. The text after it starts at 101. For grid_t = 8 (2400 tokens): t-values step 26 up to 203, and the text after starts at 205. So the text after a video depends on the video's temporal length. The HF reference matched in both cases.
- VERIFIED. The HF `t_index` for video: `arange(grid_t) * second_per_grids * 13` (`modeling_qwen3_omni_moe.py:385-386`).

**Implication for "a frame appended later shifts later text positions but not earlier ones":** TRUE for images, and it is verified by the A/B check. Per appended 640x480 frame, the position budget is VS (1) + max(15,20) (=20) + VE (1) = 22 slots. Per appended 1280x720 frame it is 1 + 40 + 1 = 42 slots. (Observed: Q moved 72 to 114, a shift of 42 = VS + 40 + VE.) The KV-token count for an image is 300 or 880, which is a different quantity from the position budget. Video is the exception: the position budget depends on the temporal length.

### Q5 (vision token count)

- VERIFIED (config). `preprocessor_config.json`: patch_size 16, merge_size 2, temporal_patch_size 2, min_pixels 3136, max_pixels 12845056 (top-level keys, not a `size` dict).
- VERIFIED (code). `image_processing_qwen2_vl.py:110-118` pops top-level min_pixels/max_pixels from the init kwargs and writes them into `size`. CPU check: `ip.size` shows shortest_edge=3136, longest_edge=12845056. So the config values are honored.
- VERIFIED (code). Resize: factor = patch x merge = 32 (`_preprocess` L238). `smart_resize` (L63-87): `h_bar = round(H/32)*32`, `w_bar = round(W/32)*32`, with a clamp to [min_pixels, max_pixels]. Grid = (1, h_bar/16, w_bar/16). Tokens = grid_t x grid_h x grid_w / 4 (the merge x merge factor; the placeholder count is `prod(grid)//4`, `omni L965-978` and `upstream L1486`).
- VERIFIED (CPU):
  - **640x480**: resized to 480x640 (no change), grid_thw = [1, 30, 40], **300 vision tokens**.
  - **1280x720**: 720/32 = 22.5, which Python `round()` takes to 22, so h_bar = 704 (not 720), w_bar = 1280. grid_thw = [1, 44, 80], **880 vision tokens**. A naive 720/16 x 1280/16 / 4 = 900 would be wrong by 20 tokens.
- VERIFIED (CPU). `AutoProcessor` (full Qwen3-Omni processor, tokenizer files only) with `<|vision_start|><|image_pad|><|vision_end|>` gives 300 image pads for 640x480, matching the image processor.

## CPU check

- Script: `/tmp/audit_tmrope/check_positions.py`.
- Command: `cd /tmp/audit_tmrope && CUDA_VISIBLE_DEVICES= /home/chaofan/workspaces/mm_reuse_study/envs/vllm-omni/bin/python -B check_positions.py > run1.log 2>&1` (exit 0).
- What it does:
  1. Loads only `preprocessor_config.json` through `AutoImageProcessor` and runs synthetic random RGB images of 640x480 and 1280x720 through it, alone and batched in both orders.
  2. Builds synthetic token sequences (sys 20, hist 30, question 10, VS=151652, image_pad=151655, VE=151653) with mm feature stubs carrying `image_grid_thw` from the processor output.
  3. Calls the vLLM-Omni thinker `get_mrope_input_positions` (bound to a stub `self` with the snapshot `thinker_config`), the upstream vLLM 0.34 implementation with the same stub, and the HF `get_rope_index` (same stub approach). Compares prefix positions for A vs B vs C.
  4. Runs video grids [4,30,40] and [8,30,40] with `second_per_grid_ts=2.0` through the same three implementations.
  5. Runs `AutoProcessor` on an image-only prompt with and without `use_audio_in_video=True`.
  6. Calls `vllm.multimodal.hasher.MultiModalHasher.hash_kwargs` with and without the flag.
- Output summary (from `run1.log`):
  - Per-image: 640x480 grid [1,30,40], 300 tokens. 1280x720 grid [1,44,80], 880 tokens.
  - Batch independence: max abs diff 0.0 in all orders.
  - Lengths A/B/C = 362 / 1244 / 352. Prefix P = 352. `A[:, :P] == B[:, :P]`: True. `A[:, :P] == C`: True.
  - Image1 t-row [51]; image2 t-row [73]; Q start A 72, B 114.
  - HF A == vLLM-Omni A: True; HF B == vLLM-Omni B: True; HF prefix A == B: True.
  - Upstream A == vLLM-Omni A and upstream B == vLLM-Omni B: True.
  - Video 4 temporal patches: t-values [21,47,73,99], Q at 101; HF match True. Video 8 temporal patches: t up to 203, Q at 205; HF match True.
  - Image-only prompt: input_ids and pixel_values identical with/without the flag: True. 300 image pads.
  - Hash: `hash(uuid only) == hash(uuid, use_audio_in_video=True)`: False. Stable across calls: True.

## Could not verify

- No engine, scheduler, or KV/prefix-cache run. Whether the identical prefix positions give real KV-cache hits depends on block hashing (which includes mm hashes) and the scheduler. I read the hash inputs but did not trace the block-hash code. UNVERIFIED.
- The 720p frame's exact shape as produced by the streaming WebSocket handler (whether it resizes or re-encodes frames before the processor). I used synthetic images as the task requested. UNVERIFIED for live frames.
- Positions with real audio items and `use_audio_in_video=True` (the audio/video pairing path). Code read only, not run. For this streaming path there are no video items, so the pairing has nothing to pair with. VERIFIED by reading, not run.
- The legacy-mode prefix-cache misses from the flag toggling (see Q3). INFERRED from code only.
- `second_per_grid_ts` values produced by the processor for real video (not used in the streaming path). UNVERIFIED.
- Rotary application itself (mrope_section / interleaved rotation) was not checked; only position ids.
- This report was written to `agents_results/`, but `agents_results/README.md` was not updated (the task said not to edit other files). The caller should add a row for this file.
