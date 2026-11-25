# Qwen3-Omni video→text: G0 source audit (M* and vLLM-Omni)

- Date: 2026-11-25 (finished 2026-11-26)
- Role: research subagent, source-level audit only. No servers started, no GPU used, no framework code changed. Two CPU-only checks were run: the HF processor was loaded in each env, and vLLM's chat-template content-format detector was run on the checkpoint's template.
- Marks: **VERIFIED** = I read the code (or ran the CPU check) that does it. **INFERRED** = follows from code I read, but the exact runtime behaviour was not executed.

## Questions (as asked)

Qwen3-Omni, VIDEO input (with audio-in-video if relevant), text output, through each framework's OpenAI chat endpoint:

1. How video becomes encoder inputs (frame sampling, temporal patching, one mm item per video or per frame), and how the video's audio is handled (use_audio_in_video or equivalent).
2. Encoder output reuse across requests: does it exist, what is the key (content hash? sampling params/fps?), granularity (whole video or frames), default capacity. For M*, any node-level or encoder caching.
3. R1′ (skip the encoder when the prefix cache already covers that media's tokens): does each framework do it? For M*, does a Thinker prefix-cache hit stop the vision/audio encoder nodes (walk scheduling vs `kv/manager.py` `apply_cached_prefix` / `resolve_cached_prefix`)?
4. Prefix caching: default on/off in the deploy configs we would use, how to turn it off (config and per request), known problems with multimodal or multi-GPU (`prefix_cache_tp`).
5. Prompt layout: where the video tokens sit relative to the system prompt and the user text. Does a system message come before the media?
6. Deploy configs that fit on 47 GB RTX 5880 Ada with at most 3 GPUs: config name, GPU count, expected memory. For vLLM-Omni: can Talker/Code2Wav be dropped for text-only output?
7. Observability: metrics for per-request encoder runs, prefix hits, per-node time; where a wrapper should hook (exact class.method) to count and time vision and audio encoder calls without changing framework code (pattern: `scripts/audit_vo/sitecustomize.py`).
8. Anything that would make S1 (same video, different questions) or S2 (same video, different system prompts) invalid, e.g. random frame sampling that changes the hash.

## Scope

- M*: `external/mstar`, commit `2c8ffde8f` (matches `externals.pins`). Env `envs/mstar` (transformers 5.24.1, torch 2.10, qwen_omni_utils 0.0.9, torchcodec 0.10).
- vLLM-Omni: `external/vllm-omni`, tag `v0.34.0` (HEAD `06a28821a`). vLLM 0.34.0 at `envs/vllm-omni/lib/python3.12/site-packages/vllm` (transformers 5.22.2, torch 2.14).
- Checkpoint: `models/cache/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/*/` (`config.json`, `preprocessor_config.json`, `chat_template.json`, safetensors headers).
- Short path names used below: `MS = external/mstar/mstar`, `VO = external/vllm-omni/vllm_omni`, `VL = envs/vllm-omni/lib/python3.12/site-packages/vllm`, `QOU = envs/mstar/lib/python3.12/site-packages/qwen_omni_utils/v2_5/vision_process.py`.
- Note: the repo was being restructured while I worked (`results/g0/*` moved under `results/g0/5880/`, `bench/g0_audit_*.py` replaced by `bench/adapters/`, `bench/audit/`). Citations into the repo's own files use the paths as they were at the end.

## Checkpoint facts (VERIFIED, from the files)

- Vision: patch 16, temporal_patch_size 2, spatial_merge 2, depth 27, deepstack layers [8, 16, 24], out_hidden 2048. `thinker_config.position_id_per_seconds` 13, `seconds_per_chunk` 2.
- `preprocessor_config.json`: `Qwen2VLImageProcessor`, min_pixels 3136, max_pixels 12845056, patch 16. Audio: WhisperFeatureExtractor, 16 kHz, 128 mel bins. There is no `video_preprocessor_config.json`.
- `generation_config.json` holds only Talker keys. The Thinker has no checkpoint sampling defaults.
- Weight sizes (bf16, from safetensors headers): Thinker LLM 56.29 GiB + lm_head 0.58, visual 1.00, audio_tower 1.21, Talker 6.2 (incl. code predictor), Code2Wav 0.40. Total 65.7 GiB. **The Thinker alone does not fit on one 48 GB card**, so both frameworks need Thinker TP2/EP2 on the 5880.
- HF processor (CPU check, both envs): `Qwen3OmniMoeProcessor` + `Qwen2VLVideoProcessor`, patch 16 / tps 2 / merge 2, `do_sample_frames=False`. Its own `size` comes from `preprocessor_config.json` (3136..12845056). Merged processor kwargs override that with `videos_kwargs.size = {shortest_edge 131072, longest_edge 786432}` (transformers `processing_qwen3_omni_moe.py:84-97` in the vllm-omni env). Example from a full processor call: 32 frames at 720x1280 give `video_grid_thw=[16,40,72]`, i.e. **11520 Thinker tokens**.

---

## 1. How video becomes encoder inputs

### M*

- **Frame sampling, VERIFIED.** `Qwen3OmniModel.load_video` (`MS/model/qwen3_omni/qwen3_omni_model.py:916-940`) calls `qwen_omni_utils.fetch_video({"video": path}, image_patch_size=14, return_video_sample_fps=True)`. Defaults (`QOU:26-34`): FPS 2.0, min 4 frames, max 768, frame count rounded to even (`smart_nframes`, `QOU:144-182`). Frames are picked deterministically with `torch.linspace(start, end, n).round()` (decord `QOU:317`, torchcodec `QOU:366`, M*'s budgeted PyAV fallback `MS/model/qwen3_omni/video_reader.py:86`). No randomness.
- **Resize, VERIFIED (the effect is INFERRED).** `fetch_video` resizes to a multiple of 28 (because `image_patch_size=14`), capped at 768·28² px per frame (`QOU:403-451`). `process_prompt` then runs `self._processor.video_processor(videos=video, size={shortest_edge 128·32·32, longest_edge 768·32·32})` (`qwen3_omni_model.py:1109-1124`), which resizes again to a multiple of 32 (patch 16). So frames are resized twice.
- **Temporal patching, VERIFIED.** tps 2 means 2 frames per temporal grid. `video_second_per_grid = temporal_patch_size / video_sample_fps` (`qwen3_omni_model.py:1110-1122`). Tokens per video = `prod(grid)/4` plus 2 sentinels (`_state_layout`, `qwen3_omni_model.py:1175-1186`).
- **Granularity, VERIFIED.** One video is one attachment and one `prefill_vision` walk. Video tensors are bound under the vision input names (`_MM_WALKS`, `qwen3_omni_model.py:872-883`). The walk is `Sequential([vision_encoder, Thinker])` (`qwen3_omni_model.py:588-622`). The vision encoder packs all frames of the video into one call, one ragged-attention segment per frame (`MS/model/qwen3_omni/submodules.py:156-226`).
- **Audio in video, VERIFIED: not supported.** `load_video` starts with `# TODO: support audio in video` (`qwen3_omni_model.py:919`). `process_prompt` extracts no audio from the video (`qwen3_omni_model.py:1107-1124`), and there is no request knob for it. The audio track is dropped. Audio reaches M* only as separate `audio_url` / `input_audio` parts (`MS/api_server/openai/adapters.py:144-154`).
- **Thinker prefill steps, VERIFIED.** One Thinker step per attachment: the attachment plus the text before it, and the last one also takes the text after it (`_thinker_steps`, `qwen3_omni_model.py:208-221`). So `[system, user: video, question]` is a single `prefill_vision` step. The Thinker chunks it at 1024 tokens (`ThinkerSubmodule.MAX_BATCH_TOKENS`, `submodules.py:566`; chunking policy `qwen3_omni_model.py:480-493`), so a multi-thousand-token video means several Thinker steps but one encoder call.

### vLLM-Omni

- **Frame sampling, VERIFIED.** The chat path decodes video with vLLM's `VideoMediaIO`. Default `num_frames=32` (`VL/multimodal/media/video.py:97`), fps unset. Backend is opencv by default (`VL/envs.py:90`). `VideoBackend.compute_frames_index_to_sample` takes `min(num_frames, total, floor(duration·fps) if fps>0)` frames with `np.linspace(0, total-1, n, dtype=int)` (`VL/multimodal/video.py:201-237`). This is deterministic. Override with `--media-io-kwargs '{"video": {...}}'` or per request with `media_io_kwargs`. A request-level fps drops the default num_frames, and vice versa (`VL/multimodal/media/video.py:76-83`).
- **Resize and patching, VERIFIED** (HF processor in the vllm-omni env, see checkpoint facts). The HF `Qwen2VLVideoProcessor` does no further frame sampling (`do_sample_frames=False`). Frames are resized to a multiple of 32 within [131072, 786432] px. tps 2.
- **Granularity, VERIFIED.** One mm item per video. A placeholder replacement per video item (`VO/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py:980-1003`). The encoder output is split per video by grid (`VO/model_executor/models/qwen2_5_omni/qwen2_5_omni_thinker.py:967-986`). One hash per item (`VL/multimodal/processing/inputs.py:83-108`).
- **Audio in video, VERIFIED: off by default, can be enabled per request.** `use_audio_in_video` is read from `mm_processor_kwargs`, default False (`qwen2_5_omni_thinker.py:156-163`). When a request sends `mm_processor_kwargs: {"use_audio_in_video": true}`, vLLM's chat content parser fetches the video URL's audio as an extra **audio** item with the same uuid (`VL/renderers/chat_utils/content_parser.py:287-295` sync, `:521-531` async). The placeholder becomes one interleaved run `<audio_start> video/audio tokens by timestamp <audio_end>` (`qwen3_omni_moe_thinker.py:873-912, 994-1003`). Video and audio are two mm items, encoded by two towers, with two cache entries. vLLM-Omni's own `_inject_audio_from_video_urls` (`VO/entrypoints/openai/serving_chat.py:1407`) is defined but never called (grep finds only its definition).
- **Cross-framework mismatch, INFERRED.** With defaults, vLLM-Omni samples 32 frames per clip. M* samples about 2 fps. Temporal positions also differ: M* uses `tps/sample_fps`; vLLM-Omni's HF processor uses `fps` from `videos_kwargs`, default 1.0, i.e. 2 s per grid, unless vLLM passes the decoded fps (not traced). Token counts are not comparable across frameworks unless both are pinned, e.g. vLLM-Omni `media_io_kwargs {"video": {"fps": 2, "num_frames": -1}}`. Even then rounding differs (M* rounds n down to even and uses linspace+round; vLLM uses floor(duration·fps) and linspace with int truncation). Log `video_grid_thw` on both sides.

## 2. Encoder output reuse across requests

### M*: none (VERIFIED)

- A grep for encoder/embedding/media caches under `MS/` finds none. The only host copies of encoder outputs are per-request rebuild copies used for preemption (`rebuild_policies`, `qwen3_omni_model.py:495-507`), and they are not shared across requests.
- The only cross-request reuse is the Thinker KV prefix cache. For a prompt with a video it covers only the text before the video (see Q3).

### vLLM-Omni: vLLM's EncoderCacheManager (VERIFIED)

- Key = `mm_feature.identifier` = the mm hash. It hashes:
  - the video's original bytes, or the decoded frames if those are smaller (`VL/multimodal/hasher.py:129-136`);
  - the frame metadata, when present (`VL/multimodal/parse.py:412-418`);
  - the model id, the modality's `media_io_kwargs` (num_frames/fps/backend) and the flat `mm_processor_kwargs` such as `use_audio_in_video` and fps (`VL/multimodal/processing/inputs.py:40-108`).

  **So the key includes the sampling params.** Same bytes with the same kwargs give the same key; changing num_frames, fps or use_audio_in_video gives a new key.
- Granularity: the whole video item, or the whole audio item. There is no per-frame reuse.
- Capacity, in encoder embeddings (not frames): `encoder_cache_size = max(scheduler.encoder_cache_size, max_tokens_per_mm_item)` (`VL/v1/core/encoder_cache_manager.py:369-374`), and `scheduler.encoder_cache_size = max_num_batched_tokens` (`VL/config/scheduler.py:350-351`).
  - `qwen3_omni_moe.yaml` stage 0 sets `max_num_batched_tokens: 32768`, so the cache holds 32768 embeddings, about 2.8 videos of 11.5k tokens.
  - `qwen3_omni_moe_thinking.yaml` does not set it. vLLM's default for GPUs under 70 GiB is 2048 for the API server or 8192 for the LLM class (`VL/engine/arg_utils.py:3036-3043`; which usage context the stage engine gets was not traced). The cache then equals one maximum video item. **INFERRED: in practice about one video is cached unless `max_num_batched_tokens` is raised.**
- Eviction is lazy LRU: unreferenced entries are freed only when space is needed (`encoder_cache_manager.py:36-40`). The worker keeps its own `encoder_outputs` dict keyed by the same hash, and `prepare_mm_inputs` skips items already in it (`VL/v1/worker/gpu/mm/encoder_runner.py:79-108`).

## 3. R1′: skip the encoder when the prefix covers the media

### M*: R1′ is moot for video, because the prefix cache never covers a video (VERIFIED)

- **Video prompts key only their leading text.** `_state_layout` sets `metadata["keyed_spans"] = {THINKER_KV: {"main": 1}}` whenever the plan has a video, with the comment "keying past it has never been run or tested" (`qwen3_omni_model.py:1199-1201`). The data worker keys only `layout.spans[:keyed_spans]` (`MS/engine/resources/kv/manager.py:1567`, `MS/api_server/data_worker.py:1056`). Test `test_a_prompt_with_a_video_keys_only_its_leading_text` (`external/mstar/test/modular/test_qwen3_omni_prefix_cache.py:441`).
- **Size of the hit (VERIFIED code; the numbers are INFERRED).**
  - The leading text is everything before `<|vision_start|>`: system turn, `<|im_start|>user\n` and any text written before the video.
  - The lookup drops the last key's page: `walk(rooted)[:len(rooted) - 1]` (`manager.py:1036`). Pages are 128 tokens (`MS/engine/resources/kv/config.py:108`; docs "128 tokens a page").
  - So the hit is at most ⌈L/128⌉−1 whole pages of an L-token leading text. M*'s default system prompt plus the user header is well under 128 tokens, which gives **0 hit tokens**. You need a leading text of more than 256 tokens to see any hit.
- **The encoder always runs.** `prefill_vision` is `Sequential([vision_encoder, Thinker])` (`qwen3_omni_model.py:588-622`). The prefix probe runs only for the Thinker node's keyed walks, inside the engine:
  - on the first chunk, `MS/engine/engine.py:1675-1698`;
  - on the unchunked path, `_skip_cached_prefix`, `engine.py:1847-1879`;
  - through `resolve_cached_prefix` / `apply_cached_prefix` (`manager.py:976-1019, 1247-1284`).

  That is after `vision_encoder` has already produced `vision_embeds`. The partial hit (leading text only) is a "partly held item", so the walk still runs and the Thinker skips only the cached leading tokens (`_answer`, `manager.py:1200-1215`). The video's KV and all text after it are recomputed on every request.
- **Images (outside scope, for reference).** An image prompt is keyed past the image, by file digest (`kv_labels`, `qwen3_omni_model.py:1212-1223`; `_digest`, `data_worker.py:1605-1630`). There the BAGEL pattern applies: the encoder runs and its output goes unused when the walk is fully cached. INFERRED from the same graph structure; not checked further.

### vLLM-Omni: R1′ is built in (VERIFIED)

- Stage 0 uses `OmniARScheduler`, which calls the upstream `Scheduler.schedule` (`VO/core/sched/omni_ar_scheduler.py:192, 1175`). It does not override `_try_schedule_encoder_inputs`: grep finds no override in `VO/`.
- Upstream `_try_schedule_encoder_inputs` (`VL/v1/core/sched/scheduler.py:2064-2258`) looks only at mm items in `[num_computed_tokens, num_computed_tokens+num_new_tokens)` via `get_mm_features_in_window` (`:2109-2113`). A prefix hit that covers a whole video item schedules no encoder for it. A hit that ends inside the item needs the output: an ECM hit if cached (`:2198`), otherwise the encoder runs.
- With `use_audio_in_video`, video and audio items share one placeholder range, so they are covered or not together (INFERRED from `_derive_audio_from_video_placeholders`, `qwen3_omni_moe_thinker.py:1024-1094`).
- **Important for S1, INFERRED from block alignment.** KV blocks are 64 tokens (`VL/config/cache.py:185`), and the block hash carries the mm hash for blocks overlapping the video (`VL/v1/core/kv_cache_utils.py:518-570`). In S1 (`[system][video][question]`) the prompts diverge right after `<|vision_end|>`. The hit stops at the last full 64-token block before the divergence, which is almost always **inside** the video. So S1 measures **R1 (ECM hit) plus R2**, not R1′. Full coverage (true R1′) needs at least 64 tokens of shared text after the video, or a multi-turn S4.

## 4. Prefix caching: defaults and switches

### M*

- **Default on (VERIFIED).** `prefix_cache: bool = True`, `prefix_cache_tp: bool = False` (`MS/engine/resources/kv/config.py:130-131`).
- **Off above one rank unless `prefix_cache_tp: true` (VERIFIED).** `keeps_prefix_cache` (`config.py:233-242`); startup logs "prefix cache off at world size N (prefix_cache_tp opts in)" (`engine.py:1471`). `prefix_cache_tp` cannot be combined with `cpu_offload_pages`, and the pool's nodes must be on one TP group (`manager.py:616-630`; `docs/mintlify/serving.mdx:167`). **Every config that fits the 5880 runs the Thinker at TP2, so add `resources: {thinker_kv: {prefix_cache_tp: true}}`, or the cache is silently off.**
- **Off in config (VERIFIED).** `resources: {thinker_kv: {prefix_cache: false}}`.
- **Off per request (VERIFIED).** Send `prefix_cache: false` as an extra body field. It passes through as model_kwargs (`adapters.py` `_passthrough`; `data_worker.py:149-152`).
- **Also off automatically for any request with audio output (VERIFIED).** The conductor forces `prefix_cache = False` when a streamed partition (the Talker) runs (`MS/conductor/conductor.py:1466-1468`; `MS/validation/resources.py:275-292`). Text-only output keeps it.
- **Media must be sent as data URLs or files (VERIFIED).** Raw tensor inputs with the prefix cache on are refused (`data_worker.py:985-987`). The chat path saves a data URL to an upload file (`MS/api_server/media_io.py:574-591`), so a normal data-URL request is fine.
- **Multi-rank caveats (VERIFIED in docs).** Above one rank, requests with media cannot be dropped, only moved to host pages (`serving.mdx:186`). Prefix counters count only on rank 0, and only with `prefix_cache_tp` (`serving.mdx:492`, `metrics.mdx:166`). A video prompt with prefix cache on is not rebuildable (`qwen3_omni_model.py:496-499`; `serving.mdx:186`).

### vLLM-Omni

- **Defaults (VERIFIED).**
  - `qwen3_omni_moe.yaml` stage 0: `enable_prefix_caching: true` (`VO/deploy/qwen3_omni_moe.yaml:40`); off on ROCm/NPU/XPU/MUSA.
  - `qwen3_omni_moe_mrv2.yaml`: off, opt-in (`:40-42`).
  - `qwen3_omni_moe_thinking.yaml`: `enable_prefix_caching: true` at pipeline level (line 7). Pipeline-level means every stage (`docs/configuration/stage_configs.md:114`).
- **Off in config (VERIFIED in docs).** Copy the YAML with `enable_prefix_caching: false`, or `--stage-overrides '{"0": {"enable_prefix_caching": false}}'` (`docs/cli/serve.md`).
- **Off per request.** There is no explicit switch; `bench/adapters/vllm_omni.py:33-35` assumes none exists. But:
  - a unique **`cache_salt`** per request prevents any cross-request match. vLLM-Omni forwards `cache_salt` (`VO/entrypoints/openai/serving_chat.py:1150-1154, 1190-1191`), and it enters the first block's hash (`VL/v1/core/kv_cache_utils.py:655-665`). VERIFIED code path; not run. The request still writes cache blocks.
  - sending `prompt_logprobs` sets `skip_reading_prefix_cache` (`VL/sampling_params.py:629-636`). This is a side effect, not a clean switch.
- **No multimodal or TP incompatibility found in vLLM-Omni code for the thinker-only path.** The only documented exceptions are platform ones (ROCm/NPU/XPU off) and the MRv2 overlay keeping it opt-in because of Talker row merging. That does not apply to the thinker-only pipeline (INFERRED).

## 5. Prompt layout

- **Template (VERIFIED by rendering the checkpoint's `chat_template.json`).**
  - A system message, if first, is rendered first: `<|im_start|>system\n…<|im_end|>\n`.
  - User content parts are concatenated in the order written. A video becomes `<|vision_start|><|video_pad|><|vision_end|>` at its position.
  - Example: `[system SYS, user [video, "Q?"]]` renders as `<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Q?<|im_end|>\n<|im_start|>assistant\n`.
  - So **different system prompts diverge before the video** in both frameworks.
- **M*, VERIFIED.**
  - `flatten_messages` keeps message and part order; `developer` becomes system (`adapters.py:120-154`).
  - `messages_from_parts` inserts the default system prompt "You are Qwen, a virtual human developed by the Qwen team…" **only when the request has no system message** (`MS/model/multimodal.py:124-145`; `qwen3_omni_model.py:1048-1053`).
  - The template is applied once and the media spans are found in the token ids (`qwen3_omni_model.py:1055-1069`). Question before or after the video stays as written.
- **vLLM-Omni, VERIFIED.**
  - vLLM's AST detector classifies this template as `"openai"` content format (CPU check with `vllm.renderers.hf._detect_content_format`, `VL/renderers/hf.py:586-602`), so parts keep their written order.
  - This differs from the BAGEL finding (image moved first), which presumably came from a `"string"`-format template.
  - vLLM-Omni injects **no** default system prompt: grep finds no "virtual human" or default_system in `VO/entrypoints` or the qwen3_omni model. Without a system message the prompt starts at `<|im_start|>user`.
  - **Send the same explicit system prompt to both frameworks.**

## 6. Deploy configs for at most 3 × RTX 5880 Ada (48 GB)

### M*

- **Use `configs/qwen3omni_thinker_tp2.yaml`, 3 GPUs (VERIFIED placement).** Thinker TP2 on ranks 0,1; audio and vision encoders on rank 0; Talker and Code2Wav on rank 2.
- Add:
  ```yaml
  resources:
    thinker_kv:
      prefix_cache_tp: true
      max_num_pages: 1024   # suggested
  ```
- `qwen3omni_thinker_ep2.yaml` has the same placement with expert parallelism (alternative).
- **Expected memory (INFERRED arithmetic).**
  - Thinker weights about 28.4 GiB per rank.
  - Thinker KV is 12 MiB per 128-token page across all heads (`qwen3omni_2gpu.yaml` comment), so about 6 MiB per page per rank at TP2 with 4 KV heads. The default 2048 pages is about 12 GiB per rank; 1024 pages is about 6 GiB.
  - Rank 0 also holds the encoders (about 2.2 GiB) plus encoder and Thinker CUDA-graph captures and workspaces (`kernel_settings` 256+128 MB).
  - Rank 0 with the default pool is roughly 43 GiB plus graphs, which is tight. With 1024 pages it is roughly 37 GiB plus graphs.
  - Rank 2: Talker 6.2 GiB + Code2Wav 0.4 GiB + Talker KV pool.
- **Configs that do not fit (VERIFIED from the YAMLs).** `qwen3omni.yaml`, `qwen3omni_2gpu*.yaml`, `qwen3omni_colocated.yaml` and `qwen3omni_pd_*` each put a whole Thinker (57 GiB) on one GPU. `qwen3omni_full_tp2.yaml` needs 4 GPUs. `qwen3omni_thinker_tp4_sp2.yaml` needs 8.
- **Talker/Code2Wav cannot be left out (VERIFIED).** Placement validation requires every node of every walk in exactly one `node_groups` entry (`MS/validation/placement.py:266-297`). For text-only requests they run no walks (`walk_plan` returns `[]`, `qwen3_omni_model.py:851-853`; docs "Text-only mode", `qwen3_omni_model.py:28-30`). They cost memory only. Co-locating the Talker on Thinker rank 1 to save a GPU is untested (INFERRED feasible only with small pools).

### vLLM-Omni

- **Use `VO/deploy/qwen3_omni_moe_thinking.yaml`, 2 GPUs (VERIFIED).**
  - `pipeline: qwen3_omni_moe_thinker_only` gives a single Thinker stage, no Talker/Code2Wav. Its comment says it works for Instruct weights when passed with `--deploy-config`. The pipeline is registered at `VO/config/pipeline_registry.py:251` and defined at `VO/model_executor/models/qwen3_omni/pipeline.py:98-115`.
  - Settings: TP2 on devices "0,1", `gpu_memory_utilization 0.9`, `enable_prefix_caching: true`, `enforce_eager: true`, `max_num_seqs: 1`, `async_scheduling: false`.
  - Default sampling: temperature 0.4, top_p 0.9, top_k 1, seed 42, repetition_penalty 1.05. Set temperature 0 per request.
- **Suggested copy of it:** `max_num_batched_tokens: 32768` (so videos are not chunked and the ECM holds more than one video); `enforce_eager: false` if graphs fit; raise `max_num_seqs` for the concurrency runs.
- **Expected memory (INFERRED).** About 28.4 GiB Thinker LLM per rank plus about 2.2 GiB towers, up to 0.9 × 48 GB of budget, leaving roughly 8-10 GiB per rank for KV and activations. Encoder profiling with a maximum-size video item could be large; cap it with `media_io_kwargs`, `mm_processor_kwargs` or `limit_mm_per_prompt` if startup OOMs.
- **The default `qwen3_omni_moe.yaml` does not fit.** Stage 0 puts the whole Thinker on device "0". A 3-GPU variant of the full pipeline (stage 0 TP2 on 0,1; stages 1+2 on 2) would likely fit but is unnecessary.
- With the full pipeline, the chat default output modalities are all of the engine's (text+audio) (`serving_chat.py:821-823`), so text-only requests must send `"modalities": ["text"]`. With the thinker-only pipeline, text is the only output.
- Model runner defaults to v2 (MRv2) (`VO/config/stage_config.py:932`); the YAML comment says v1 overrides are unsupported with vLLM 0.34.

## 7. Observability and hook points

### M*

- **Metrics (VERIFIED), with `--enable-metrics`.**
  - `mstar_step_seconds{component, graph_walk}` histogram, one sample per engine step per node and walk (`MS/observability/catalog.py:87-97`; `metrics.mdx:159-160`). Its `_count` delta gives encoder runs: `component="vision_encoder"`, walk `prefill_vision`. Expect one vision step per video and about ⌈tokens/1024⌉ Thinker `prefill_vision` steps, because of chunking.
  - `mstar_prefix_cache_query_tokens_total` / `_hit_tokens_total{component}`, counted once per request in `_lease` (`manager.py:1051-1055`), rank 0 only.
  - All of these are aggregate counters. Per-request values need concurrency 1 and before/after scrapes with the 2 s settle (DESIGN 3.5).
- **Hook for counting and timing (VERIFIED as the common path).** `mstar.engine.engine.Engine.exec(self, batch)` (`MS/engine/engine.py:1893`). Use `batch.node_name`, `batch.step_context.graph_walk` and `batch.request_ids`. Filter `node_name in {"vision_encoder", "audio_encoder"}` and sync the device around it in time mode. Both the captured-graph replay and the eager forward go through it (`Engine._forward`, `engine.py:2497-2537`).
- **Submodule-level hooks are a weaker alternative.** `mstar.model.qwen3_omni.submodules.VisionEncoderSubmodule.forward_batched` and `mstar.model.components.aut_submodule.AuTEncoderSubmodule.forward_batched` are bypassed when the engine replays a whole-node graph lease (`cg_runner.run_forward`, `engine.py:2513-2521`). Whether the encoders take that path was not traced.
- M* workers are separate processes. A `PYTHONPATH` sitecustomize reaches them only if the launcher passes the environment through (INFERRED; it worked for the BAGEL counts via metrics, not a shim).

### vLLM-Omni

- **Metrics (VERIFIED), need `--log-stats`.** `vllm:prefix_cache_queries/hits` with stage label (`VL/v1/metrics/loggers.py:662-680`; `VO docs/usage/metrics.md:10,17`). `vllm:mm_cache_*` counts the frontend processor cache, not the encoder cache (`loggers.py:714-725`). **Per-request prefix hit:** `usage.prompt_tokens_details.cached_tokens` when the server runs with `--enable-prompt-tokens-details` (`serving_chat.py:3017-3022`); VERIFIED code path, flag availability on the omni CLI INFERRED. Per-request encoder time: `--enable-mm-processor-stats` makes the MRv2 `EncoderRunner` record `encoder_forward_secs` and `num_encoder_calls` per request (`VL/v1/worker/gpu/mm/encoder_runner.py:174-194`; `VL/v1/worker/gpu/model_states/interface.py:191-207`). It is read through `collective_rpc("get_encoder_timing_stats")`, which no HTTP route exposes (INFERRED). That makes the shim the practical route.
- **Hooks for an extended `scripts/audit_vo/sitecustomize.py`:**
  - **`vllm.v1.worker.gpu.mm.encoder_runner.EncoderRunner.execute_mm_encoder(self, mm_kwargs)`.** `mm_kwargs` is a list of `(modality, item)`. It covers both the encoder-CUDA-graph path and the eager `embed_multimodal` path (`encoder_runner.py:150-172`). Log modality counts ("video"/"audio"/"image") and time with a device sync. Encoder CUDA graphs are off by default anyway (`cudagraph_mm_encoder=False`, `VL/config/compilation.py:609`; nothing in `VO/` turns them on). VERIFIED.
  - **Per-modality alternative:** `vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerForConditionalGeneration._process_video_input` / `_process_image_input` (inherited from `Qwen2_5OmniConditionalGenerationMixin`, `qwen2_5_omni_thinker.py:951-986`) and `_process_audio_input` (`qwen3_omni_moe_thinker.py:1140-1151`). Patch the attribute on the Thinker class. `embed_multimodal` is at `qwen3_omni_moe_thinker.py:1392`; the top-level `vllm_omni...qwen3_omni.Qwen3OmniMoeForConditionalGeneration.embed_multimodal` delegates to it (`qwen3_omni.py:670-672`). VERIFIED.
  - **Keep** `vllm.v1.core.encoder_cache_manager.EncoderCacheManager.check_and_update_cache`. It runs in the scheduler process.
  - Optional scheduler-side prefix hit per request: `vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks(self, request, …)`, which returns `(blocks, num_computed_tokens, …)` (`VL/v1/core/kv_cache_manager.py:288-290`).
- **Caveats for the shim (INFERRED).**
  - (a) At TP2, each TP worker runs the towers, so expect one encoder record per rank (distinct pids). Count per pid.
  - (b) When a video is split across scheduler steps by chunked prefill (likely with the thinking YAML's default `max_num_batched_tokens`), later steps of the **same** request call `check_and_update_cache` and log `hit=True`. Treat only the first check per (request, input_id) as the cross-request ECM result.
  - (c) The startup profile runs dummy encoder calls; count only inside the request window (DESIGN 3.5).
  - (d) `bench/adapters/vllm_omni.py` classifies "no ecm_check and no encoder call" as full coverage. That still holds, because the scheduler checks the ECM only for items in the window.

## 8. Threats to S1 / S2 validity

- **Frame sampling is deterministic in both frameworks (VERIFIED, linspace, no RNG).** Identical bytes with identical kwargs give the same vLLM mm hash and the same M* file digest. Changing per-request `media_io_kwargs`, `mm_processor_kwargs` (fps, num_frames, `use_audio_in_video`) changes the vLLM hash and breaks reuse. Keep them fixed across a sequence.
- **M*: S1 and S2 cannot show any video reuse (VERIFIED).** The prefix cache keys only the leading text before a video, the hit is whole 128-token pages minus the last, and there is no encoder cache. With a short system prompt, S1 gets 0 hit tokens and runs the vision encoder plus the full Thinker prefill every time, the same as S3. **For M*, S1/S2 measure the baseline, i.e. the upper bound of what I1/I2 plus video keying could save.**
- **vLLM-Omni S1 is mostly R1 (ECM hit) plus partial R2, not R1′ (INFERRED, see Q3).** Measuring true R1′ needs at least 64 tokens of common text after the video, or a multi-turn S4. S2 diverges before the video, so the hit is roughly 0 and the run depends on the ECM (R1).
- **vLLM-Omni encoder cache capacity (INFERRED).** With the thinking YAML defaults the ECM holds about one maximum video. Any interleaving of a second large video between S1 requests can evict the first. Set `max_num_batched_tokens` explicitly and log ECM checks.
- **No default system prompt in vLLM-Omni; M* adds one when absent (VERIFIED).** Always send an explicit system message, also so M* prompts don't silently get the "virtual human" persona.
- **Audio-in-video (VERIFIED).** It is available only in vLLM-Omni (request `mm_processor_kwargs`), and it adds an audio item and changes the video hash factors. M* cannot do it, so cross-framework comparisons must use `use_audio_in_video=false`, or silent clips.
- **Sampling defaults differ.** M* Thinker falls back to temperature 0.7 / top_p 0.9 (`qwen3_omni_model.py:144-158`; the checkpoint has no Thinker keys). The vLLM-Omni thinking YAML uses temperature 0.4, top_k 1, repetition 1.05. Send temperature 0 (and the same max_tokens) to both. VERIFIED.
- **Token counts differ across frameworks by default (INFERRED, see Q1).** 32 frames vs 2 fps, plus a double resize in M*. Report savings within each framework only, or pin the frame counts and record the grids.
- **M* metrics at TP2 count only with `prefix_cache_tp` on.** Without it, hit/query stay 0 because the cache is off, which an audit could misread as "no hit" (VERIFIED, `engine.py:1471`).

## Could not verify

- Actual GPU memory of either config on the 5880 (not started). Whether M*'s default 2048-page Thinker pool plus captures fit on rank 0. Whether vLLM-Omni's encoder profiling with a maximum video fits.
- Whether M*'s encoder nodes ever run through a whole-node CUDA-graph lease that skips `forward_batched` (it decides between the `Engine.exec` and submodule hooks).
- The usage context, and so the default `max_num_batched_tokens`, that vLLM-Omni stage engines get when the YAML omits it.
- Whether vLLM passes the decoded video fps into the HF processor's `fps` (affects temporal positions and `second_per_grid`, not token count or hash).
- That vLLM's TP-sharded vision and audio towers in vLLM-Omni log one call per rank. Expected from vLLM's design; not read line by line.
- Whether `--enable-prompt-tokens-details` is accepted by `vllm serve --omni` (vLLM-Omni's serving_chat reads the attribute; CLI wiring not traced).
- The precise layout of the full 3-stage vLLM-Omni pipeline for text-only requests (whether stage 1 is ever touched with `modalities: ["text"]`). Not needed if the thinker-only pipeline is used.
