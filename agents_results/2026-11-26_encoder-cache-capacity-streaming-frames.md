# Encoder cache capacity and eviction for the streaming frame path (Qwen3-Omni, stage 0)

## Question(s) asked

- Q1. Which model runner and scheduler run stage 0 under the default deploy YAML, which ECM code path applies, and which usage context and `max_num_batched_tokens` the stage engine gets when the YAML omits it.
- Q2. Encoder cache size in embeddings for stage 0 under the four Qwen3-Omni deploy YAMLs, and frames that fit at 640x480 and 1280x720.
- Q3. Eviction behaviour for the streaming pattern (warmups reference all buffered frames): (a) re-encode of cached frames, (b) eviction order and whether a buffered frame can be evicted and re-encoded, (c) abort handling; plus the chunked-prefill re-check case.
- Q4. Can the encoder cache hold a 50-frame buffer at 640x480 and 1280x720? If not, how many frames fit and the re-encode cost per warmup. KV-side order of magnitude.

## Scope (commits / paths)

- vLLM-Omni: `/home/chaofan/workspaces/mm_reuse_study/external/vllm-omni`, tag v0.34.0, HEAD `06a28821a7f72bdb436c3df656b10f67fded0846`. Read-only.
- vLLM 0.34 installed: `/home/chaofan/workspaces/mm_reuse_study/envs/vllm-omni/lib/python3.12/site-packages/vllm`.
- Checkpoint config: `/home/chaofan/workspaces/mm_reuse_study/models/cache/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695/`.
- Deploy YAMLs: `vllm_omni/deploy/qwen3_omni_moe.yaml`, `qwen3_omni_moe_thinking.yaml`, `qwen3_omni_moe_mrv2.yaml`, `qwen3_omni_duplex_mrv2.yaml`.
- Runs: CPU only (`CUDA_VISIBLE_DEVICES=` set), no servers, no GPU, no edits to the repo or vLLM. Scratch scripts in `/tmp/audit_ecm/`: `hf_tokens.py`, `vllm_budget.py`, `engine_defaults.py` (per-YAML cases), `debug_budget.py`, `mro_check.py`, `mro2.py`, `vllm_frame_tokens.py`, `weights.py`, `sim_ecm.py` (warmup simulation on the real `EncoderCacheManager`), `cases.py` (deterministic eviction, abort, chunked cases).

## Corrections to the brief's premises

1. **The video term sets the ECM size, not the image term.** `compute_mm_encoder_budget` takes the max over all active modalities (`encoder_cache_manager.py:356`), and video is active by default (allowed limit 999). The vLLM-Omni Qwen3-Omni processing info gives a video max of 62720 tokens, larger than the image max of 12544. So the ECM is 62720 for every YAML, not max_pixels-based.
2. **Warmups are serialized and coalesced.** A frame arrival does not start a second warmup while one is in flight (`video_stream_base.py:466-478`). Frames that arrive during a warmup are picked up by the next one (`:511-517`). Only a query cancels a warmup (`:521-533`, `:553-557`). So the number of warmups is at most the number of frames.
3. **There is no V1 `vllm/v1/worker/gpu_model_runner.py` in vLLM 0.34.** The installed `vllm/v1/worker/` has `gpu/` (MRv2) and `gpu_worker.py`, but no V1 `gpu_model_runner.py`. vLLM-Omni rejects V1 (`vllm_omni/engine/arg_utils.py:484-485`, `vllm_omni/config/stage_config.py:1369-1374`).
4. **The YAML header comment is wrong.** `qwen3_omni_moe.yaml:14-15` says stages 0/1 "fit in the 32768 default". No such default exists in code. The 32768 is the explicit stage-0 value at `qwen3_omni_moe.yaml:35`. Without it, the default depends on hardware (see Q1).
5. **`SchedulerConfig.encoder_cache_size` is only a floor.** The scheduler uses `mm_budget.encoder_cache_size` (`v1/core/sched/scheduler.py:282-294`), which comes from `MultiModalBudget` (`multimodal/encoder_budget.py:119-125`), which takes the max over modalities.

## Findings per question

### Q1. Runner, scheduler, ECM path, usage context, default MNBT

- **Model runner (VERIFIED):** `model_runner: v2` at the top of `qwen3_omni_moe.yaml:22`, with no stage override for stage 0. The CUDA platform block (`qwen3_omni_moe.yaml:95-99`) only sets `compilation_config`; musa, npu and rocm overrides are at `:104`, `:116`, `:142`. `use_v2_model_runner` defaults to True (`vllm_omni/engine/arg_utils.py:346`). The MRv2 worker is `vllm/v1/worker/gpu/model_runner.py`; its `EncoderCache` is created at `gpu/model_runner.py:284-287` for the first PP rank. vLLM-Omni's `worker_v2/omni_model_runner.py:470-476` sets `encoder_cache = None` only when `has_preprocess` is set, and in Qwen3-Omni that is set only on the talker stage (`model_executor/models/qwen3_omni/qwen3_omni.py:330`). Stage 0 (thinker) keeps its encoder cache.
- **Scheduler (VERIFIED for class hierarchy; INFERRED for runtime class):** `OmniARScheduler(OmniSchedulerMixin, VLLMScheduler)` at `vllm_omni/core/sched/omni_ar_scheduler.py:192`, and `OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler)` at `:2730`. Neither overrides `_try_schedule_encoder_inputs`, `check_and_update_cache` or `can_allocate` (grep for `encoder` shows only `free`/`_free_encoder_inputs` call sites). Class choice: `vllm_omni/config/stage_config.py:1654-1659` uses `engine_args.get("async_scheduling", True)` to pick `OmniARAsyncScheduler`. Stage 0 does not set `async_scheduling` in any of the four YAMLs, so the async class is INFERRED; the runtime class was not checked in a log. The ECM logic is vLLM's either way.
- **ECM path (VERIFIED):** scheduler-side `EncoderCacheManager` (`v1/core/encoder_cache_manager.py:18`), created at `v1/core/sched/scheduler.py:293-294`. The enc-dec variant is not used (`is_encoder_decoder` is False for Qwen3-Omni). The EC connector is off (`config/ec_manager_config.py:25-28`, `cls_path is None` returns None). Worker-side: `v1/worker/gpu/mm/encoder_cache.py:15` (`encoder_outputs: dict[str, Tensor]` keyed by mm hash). The worker skips encoding when the hash is present (`v1/worker/gpu/mm/encoder_runner.py:92-93`) and drops entries only when the scheduler lists them as freed (`v1/worker/gpu/model_runner.py:1272-1274`, from `encoder_cache_manager.py:312-326`).
- **Usage context (VERIFIED):** the stage engine config is built with `UsageContext.LLM_CLASS` (`vllm_omni/engine/stage_init_utils.py:1649-1651`). vLLM's `create_engine_config` calls `_set_default_max_num_seqs_and_batched_tokens_args` (`engine/arg_utils.py:2676`, defined `:3186-3212`) using `get_batch_defaults` (`:2977`).
- **Default MNBT when the YAML omits it, for LLM_CLASS (VERIFIED from code, and by CPU run with mocked device memory):**
  - device >= 160 GiB: 16384 (`arg_utils.py:3012-3014`).
  - 70 GiB <= device < 160 GiB, not A100, not GB10: 16384 (`:3026-3028`). This covers H100 80 GB and H200 141 GiB.
  - smaller devices: 8192 (`:3036-3038`). This covers a 48 GB card.
  - CPU run results: H200 141 GiB omitting MNBT gives `max_num_batched_tokens=16384`; H100 80 GiB gives 16384; a 48 GiB card gives 8192 (`/tmp/audit_ecm/engine_defaults.py`).
  - `qwen3_omni_moe_thinking.yaml` omits it (`:10-12`), so it gets 16384 on H100/H200 and 8192 on 48 GB.
  - `qwen3_omni_moe_mrv2.yaml:18` and `qwen3_omni_duplex_mrv2.yaml:5` inherit base `qwen3_omni_moe.yaml` (32768).
- **ECM sizing chain (VERIFIED):** `cache = max(scheduler_config.encoder_cache_size, max_tokens_per_mm_item)` (`encoder_cache_manager.py:372-374`), with `scheduler_config.encoder_cache_size = max_num_batched_tokens` (`config/scheduler.py:350-351`), and `max_tokens_per_mm_item = max(active modality maxima)` (`encoder_cache_manager.py:356`). Budget: `encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)` (`:369-371`). Active modalities: allowed limit > 0 (`encoder_budget.py:73-85`, `:106-109`).

### Q2. ECM size and frames that fit

- **Max image tokens (VERIFIED, code + CPU run):** `max_pixels // (32*32) = 12845056 // 1024 = 12544`. The factor pair is 112x112, so the largest feature-maximising image is 3584x3584 px, giving 12544 tokens (`qwen2_vl.py:1079-1111`, `:1130-1138`; CPU run `vllm_frame_tokens.py` returns 12544). HF processor check (`hf_tokens.py`): 3584x3584 gives grid [1,224,224] and 12544 tokens.
- **Max video tokens (VERIFIED, engine path):** the vLLM-Omni fork's `Qwen3OmniMoeThinkerProcessingInfo(Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo)` (`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py:762`) resolves `get_max_video_tokens` to `Qwen2VLProcessingInfo` (`vllm/model_executor/models/qwen2_vl.py:1179-1190`). It is frame-based: frames at 3584x3584 that fit `seq_len = max_model_len = 65536` (`:1141-1162`). That is 10 frames (5 temporal pairs x 12544 = 62720), capped by `_MAX_FRAMES_PER_VIDEO = 14` (`:120`, not binding). Result: 62720 (`/tmp/audit_ecm/mro2.py`, `vllm_budget.py`).
- **Note on the upstream class:** `mro_check.py` shows that when vLLM-Omni's plugin is not loaded first, the registry returns upstream's `Qwen2_5OmniThinkerProcessingInfo`, whose video max is `seq_len` = 65536. The real engine path loads the vLLM-Omni class, so 62720 is the operative number. The conclusions do not change at 65536.
- **Max audio tokens:** 750 (`engine_defaults.py` output).
- **max_model_len (VERIFIED):** 65536 with no vLLM-Omni override for stage 0 (from `text_config.max_position_embeddings`; `vllm_budget.py`).
- **M = max(12544, 62720, 750) = 62720.** The ECM is `max(MNBT, 62720)` and the compute budget is `max(MNBT, 62720)`.

| YAML (stage 0) | MNBT used | ECM size (embeddings) | Source |
|---|---|---|---|
| `qwen3_omni_moe.yaml` (default) | 32768 explicit (`:35`) | **62720** | `engine_defaults.py` "default YAML" case: `encoder_cache_size=62720` |
| `qwen3_omni_moe_thinking.yaml` | 16384 on H100/H200; 8192 on 48 GB (default, omitted) | **62720** | `engine_defaults.py` "omits MNBT" cases |
| `qwen3_omni_moe_mrv2.yaml` | 32768 (inherited base) | **62720** | inherits base (`:18`) |
| `qwen3_omni_duplex_mrv2.yaml` | 32768 (inherited base) | **62720** | `engine_defaults.py` "duplex limits" case, video still active at 999 |

Caveat: in the duplex case the SchedulerConfig field reads 32768, but `MultiModalBudget` gives 62720, and the scheduler uses the latter (`scheduler.py:285`).

- **Frames that fit (ECM = 62720):**
  - 640x480: 300 tokens per frame (HF and vLLM agree: 300). 62720 / 300 = 209.07, so **209 frames**. A 50-frame buffer is 15000 embeddings (24% of the cache).
  - 1280x720: the processor resizes to **1280x704** (`smart_resize`, factor 32; HF run), giving 880 tokens (vLLM agrees: 880). 62720 / 880 = 71.27, so **71 frames**. A 50-frame buffer is 44000 embeddings (70%).
  - The streaming path does not resize frames before the processor (no resize in `video_stream_base.py` or `serving_video_stream.py`, grep empty), so the client resolution reaches the processor.
- **Sensitivity only (not what the code gives):** if the ECM were 32768 (MNBT alone): 480p fits 109 frames, 720p fits 37 frames.

### Q3. Eviction behaviour

- **Data structures (VERIFIED):** `cached: mm_hash -> set(request_ids)`, `num_embeds_by_hash`, `freeable: OrderedDict[mm_hash -> embeds]` (`encoder_cache_manager.py:82-94`). An entry is in `freeable` only when no request references it.
- **check_and_update_cache (`:111-139`):** if the hash is not in `cached`, return False (miss). If it is cached and currently unreferenced, pop it from `freeable` (`:133-135`). Add the request to the set (`:137`), record the input id (`_add_cached_input_id`, `:245-253`, guarded by `if input_id in cached_ids: return` at `:249-250`), and return True.
- **can_allocate (`:148-209`):** fits in free slots (`:193-194`), or evicts from `freeable` front (`:203-208`: `popitem(last=False)`, delete from `cached` and `num_embeds_by_hash`, append to `freed`) until the request fits.
- **allocate (`:211-243`):** only after `can_allocate` returned True (asserts at `:237-238`).
- **free_encoder_input (`:259-298`):** removes this request's reference. If the request still holds the hash through another input (`:292-293`), it keeps it. When the set becomes empty, the entry is appended to `freeable` at the tail (`:294-298`).
- **free (`:300-310`):** frees every cached input of a request. **Called on finish, cancel/abort, and preemption** (`scheduler.py:1872`, `:3191`), and when a request cannot be scheduled without chunked prefill (`:1453`, `:1522`).
- **Scheduler call sites (VERIFIED):** `_try_schedule_encoder_inputs` (`scheduler.py:2064`): the hit check at `:2198` (`if check_and_update_cache(...): continue`), then `can_allocate` at `:2203`. `_allocate_encoder_inputs` (`:2050-2062`) calls `allocate` for new items and `check_and_update_cache` for duplicates in the same step. Items are released after the step runs, when they are past `num_computed_tokens` (`:2425-2437`, `_free_encoder_inputs` at `:2961-2988`).

**(a) Is a cached frame re-encoded when a later warmup references it?** No. The hit path returns True (`encoder_cache_manager.py:127-139`). The scheduler skips scheduling it (`scheduler.py:2198-2201`), and the worker skips it because the hash is already in `encoder_outputs` (`encoder_runner.py:92-93`). Simulated: a hit costs no encode (`cases.py` Case A, Case B `W2: f1 HIT`).

**(b) Eviction order and whether a buffered frame can be evicted and re-encoded.**
- Victims are unreferenced entries only, oldest-released first (`freeable` front). Because a hit pops the entry and a later release appends it to the tail, this is LRU over unreferenced entries (VERIFIED from code, `:133-135`, `:294-298`, `:203-208`).
- Yes, a frame still in the session buffer can be evicted, but only when unreferenced and the cache is full. Simulated (`cases.py` Case B, cap 3, buffer 4): W3 re-encodes f0, f1, f2 after they were evicted, and f3 cannot allocate because W3's own f0..f2 are referenced.
- At default capacity (62720) it does not happen in steady state: the referenced set is at most the 50-frame buffer, which fits in 71 slots at 720p. Stale frames dropped from the buffer (the oldest 25 on each drop, `video_stream_base.py:659-675`) sit in `freeable` ahead of the current buffer frames, so they are evicted first. Simulation of 200 warmups at 62720: 1.00 new encodes per warmup, 37 hits (720p); 1.00 at 480p (`sim_ecm.py`).

**(c) Aborted warmup.** The abort path is `_cancel_warmup` -> `engine_client.abort(request_id)` (`video_stream_base.py:521-533`) -> `finish_requests` (`scheduler.py:3100-3167`) -> `_free_request`, which calls `encoder_cache_manager.free(request)` (`:3191`). That releases the references. Entries are not deleted: they become freeable and stay cached until evicted (`cases.py` Case C: after abort, `freeable=[f0,f1]`, and the next request hits f0 with no encode). Worker tensors are dropped only when the scheduler reports the hash as freed (`model_runner.py:1272-1274`).

**Chunked prefill.** Yes, the scheduler re-checks an item when a later step's window overlaps it (`scheduler.py:2198` is called every time). If the same request still references it, `check_and_update_cache` returns True without double-counting (`encoder_cache_manager.py:137-138`, `:249-250`; `cases.py` Case D: second check returns True, refcount set is `{'R'}`). A referenced entry is not in `freeable`, so it cannot be evicted between steps. Items are released only after the step runs and the request has passed them (`scheduler.py:2425-2437`).

### Q4. Can the cache hold a 50-frame buffer, and re-encode cost

- **Buffer policy (VERIFIED):** `max_frames` default 50 (`video_stream_base.py:165-168`). On the incremental path, at the cap, the oldest `max_buf // 2 = 25` frames are dropped (`:659-675`). So the buffer oscillates between 26 and 50 frames (mean 38 in simulation).
- **Fit at ECM 62720:** 50 frames fit at both resolutions (15000 at 480p; 44000 at 720p, each under 62720). Headroom: 209 frames at 480p, 71 frames at 720p.
- **Re-encode per warmup (VERIFIED by simulation, INFERRED for real traffic):** steady state is 1 newly encoded frame per warmup (the new frame), with the rest as hits. The simulation models one new frame per warmup, which is the worst case for warmup count; coalescing reduces warmups but not per-warmup misses.
- **Sensitivity at ECM 32768 (not the code's value):** at 720p, about 22 of 38 frames are re-encoded per warmup (21.88 mean, `sim_ecm.py`). At 480p, 1.00 (109 frames fit).
- **Prefix (KV) side:**
  - Thinker KV per token = 2 x 48 layers x 4 KV heads x 128 dim x 2 bytes (bf16) = 98304 B = 96 KiB (`config.json` text_config). The bf16 dtype is INFERRED from the checkpoint size; the KV dtype default was not traced.
  - Thinker weights = 59.08 GiB (bf16, summed from safetensors headers by `weights.py`).
  - H100 80 GB at `gpu_memory_utilization=0.9`, assuming about 79 GiB usable: 0.9 x 79 = 71.1 GiB; minus 59.1 = 12.0 GiB; minus an assumed 2-4 GiB overhead gives 8-10 GiB of KV, or about **87k-109k tokens (order 10^5)**. The overhead term is UNVERIFIED.
  - H200 141 GiB: 0.9 x 141 = 126.9; minus 59.1 = 67.8; minus an assumed 2-4 GiB gives about 64-66 GiB, or about **0.70M-0.72M tokens (order 10^6)**.
  - A 50-frame warmup prompt is about 15k tokens (480p) or 44k tokens (720p). On an 80 GB card that is roughly half the KV budget at 720p.
  - Prefix blocks are chained from the start of the prompt (`v1/core/kv_cache_utils.py:703-728`, `parent_block_hash`). When the drop removes the first 25 frames, every later block hash changes, so the KV prefix misses from the first frame on that warmup. The encoder cache still hits, since it is keyed by mm hash. INFERRED from the chained hash, not run.

## Arithmetic table

| Quantity | Value | Derivation | Mark |
|---|---|---|---|
| Image max tokens | 12544 | 12845056 // 1024; 112x112 merged grid = 3584x3584 px | VERIFIED |
| Video max tokens (engine) | 62720 | 65536 seq_len -> 10 frames at 3584x3584 = 5 x 12544 | VERIFIED |
| Audio max tokens | 750 | Qwen2Audio max | VERIFIED |
| ECM size, all four YAMLs | 62720 | max(MNBT, 62720), MNBT in {32768, 16384, 8192} | VERIFIED |
| Tokens per 640x480 frame | 300 | grid [1,30,40]: 30*40/4 | VERIFIED (HF and vLLM) |
| Tokens per 1280x720 frame | 880 | resized 1280x704 -> grid [1,44,80]: 44*80/4 | VERIFIED (HF and vLLM) |
| Frames fit, 480p | 209 | floor(62720/300) | VERIFIED arithmetic |
| Frames fit, 720p | 71 | floor(62720/880) | VERIFIED arithmetic |
| 50-frame buffer, 480p / 720p | 15000 / 44000 | 50 x tokens | VERIFIED arithmetic |
| Sensitivity (ECM 32768): frames fit 480p / 720p | 109 / 37 | floor(32768/tokens) | VERIFIED arithmetic |
| Sim: new encodes per warmup, ECM 62720, 720p / 480p | 1.00 / 1.00 | `sim_ecm.py`, last 100 of 200 | VERIFIED (model) |
| Sim: new encodes per warmup, ECM 32768, 720p | 21.88 | `sim_ecm.py` | VERIFIED (model) |
| KV per token | 96 KiB | 2 x 48 x 4 x 128 x 2 B | VERIFIED arithmetic; bf16 INFERRED |
| Thinker weights | 59.08 GiB | safetensors headers | VERIFIED |
| KV tokens, H100 80 GB | ~87k-109k | see Q4 | INFERRED (overhead UNVERIFIED) |
| KV tokens, H200 141 GiB | ~0.70M-0.72M | see Q4 | INFERRED (overhead UNVERIFIED) |

## Could not verify

- **No live run.** All engine-side numbers come from a CPU-only engine config (`create_engine_config` + `MultiModalBudget`) with mocked device memory. A live stage log would show `Encoder cache will be initialized with a budget of N tokens` (`encoder_runner.py` `profile_encoder_cache`). Expected N = 62720.
- **Runtime scheduler class** (OmniARAsyncScheduler vs OmniARScheduler) for stage 0 was not confirmed in logs. It does not change the ECM logic.
- **Simulation fidelity:** `sim_ecm.py` is a frame-level model (immediate allocation, chunk = MNBT // tokens_per_frame frames per step, release of a step's frames after the step). The real scheduler works per token with `num_embeds_to_schedule` accounting. Hit/miss counts are model-level, not traced from a live engine.
- **Abort mid-step:** the path where an item is allocated in step N but the abort lands before the worker runs it was not traced. INFERRED that the scheduler's cached state would still reference the entry until `free`.
- **Processor (sender) cache interaction** with re-encode after eviction. `encoder_cache.py:29-35` says EngineCore keeps the original payload for replay, which suggests re-encode works, but the sender-cache path was not traced (`mm_processor_cache_gb` and `video_stream_base.py:1148`).
- **KV overhead and KV dtype:** the overhead term (2-4 GiB) and the default KV dtype were not derived from code. The KV numbers are order-of-magnitude only.
- **Warmup timing:** arrival-to-warmup timing (how many frames coalesce per warmup) was not measured. The simulation uses one new frame per warmup.
- **5880 hardware:** the 48 GB card example is an assumption. A 48 GB card cannot hold the 59 GiB thinker at 0.9 utilisation in bf16, so any 5880 run uses a different configuration; not verified.
- **README index:** `agents_results/README.md` was not updated, since the instruction was to edit no other files.
