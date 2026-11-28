# Measurement hooks: process model, num_cached_tokens, warmup abort, encoder timing

## Question(s) asked

Read-only source audit for a study that separates, per query, (a) encoder run, (b) prefill progress, (c) cache hit, for the WebSocket streaming video handler (`video_stream_base.py`) on Qwen3-Omni stage 0.

- Q1. Process model for `vllm serve <Qwen3-Omni> --omni`: which processes run the scheduler (EncoderCacheManager, KV block pool) and which run the model worker (vision encoder). Spawn or fork? Does a PYTHONPATH `sitecustomize.py` load in each?
- Q2. `num_cached_tokens` path from the scheduler to `OmniRequestOutput`, for MRv2, V1, async_chunk stage 0, and max_tokens=1. Is it reachable in `_drive_warmup_generate` and `_drive_query`? Least-change way to get a client- or log-facing value.
- Q3. Abort semantics for warmup requests: which full blocks stay reusable by the query; partial last block; omni-specific abort paths.
- Q4. Engine-side encoder timing hooks: per-item or per-batch, request id available, process; recommend one hook.
- Q5. `--enable-mm-processor-stats`: wired into `vllm serve --omni`? What it records, where exposed, attributability of warmup ids.

## Scope

- vLLM-Omni: `/home/chaofan/workspaces/mm_reuse_study/external/vllm-omni` (tag v0.34.0, HEAD `06a28821a`). Read only.
- vLLM 0.34: `/home/chaofan/workspaces/mm_reuse_study/envs/vllm-omni/lib/python3.12/site-packages/vllm` (installed copy). Read only.
- Stdlib: `/usr/lib/python3.12/multiprocessing/`, `/usr/lib/python3.12/subprocess.py` (the env python symlinks to `/usr/bin/python3.12`).
- One scratch experiment (no vLLM import, no GPU, no server): `/tmp/audit_meas/spawn_test.py`, `/tmp/audit_meas/sc/sitecustomize.py`.
- Not modified: any file under `external/` or the vLLM install. No servers, no GPU.
- Prior notes read first (not redone): `2026-11-25_qwen3-omni-g0-code-audit.md`, `2026-11-26_vllm-omni-streaming-input-audit.md` (sections 3b and 4).

Marks: **VERIFIED** = code read or scratch experiment run. **INFERRED** = follows from code read; not executed. **UNVERIFIED** = not traced.

---

## Summary of answers

1. **Process model.** One API/orchestrator process, then one spawned process per stage replica. For stage 0 at TP=1, the scheduler, EncoderCacheManager, KV block pool, and the MRv2 model runner (vision encoder) are all in the same spawned `StageEngineCoreProc` process. Spawn, not fork. The environment is inherited, so a PYTHONPATH `sitecustomize.py` loads in the parent and in each spawned child (VERIFIED by scratch experiment). It does not load if the server is started with `-I` (or `-S`).
2. **num_cached_tokens.** Set at first scheduling (`PrefillStats`), emitted on the first output step that is emitted. Populated for MRv2 (the only runner in 0.34 for omni; V1 is refused). Caveats: (i) an output with no prefill stats reads `0`, which is indistinguishable from a miss; (ii) for MRv2 + async_chunk + non-final stage, a non-stopping first step drops the stats; (iii) `_drive_warmup_generate` discards every output (`async for _ in result_gen: pass`), so warmups aborted before their first token report nothing. The scheduler-side prefix-hit count at admission is the reliable source.
3. **Abort.** Every full block that was hashed when scheduled stays in the prefix-cache hash map after the abort (freed to the tail of the LRU free queue) until evicted. Partially computed last blocks are not reusable by default. Omni adds no abort-side release of its own on the KV path. The omni prefix cache (Qwen3-Omni MRv2 only) forces pending writes to land on abort.
4. **Encoder hooks.** The best single hook is `vllm.v1.worker.gpu.model_states.interface.ModelState.execute_mm_encoder(self, scheduled_encoder_inputs)`. It is per step, has engine request ids as dict keys, and runs in the worker (= the stage process at TP=1). `EncoderRunner.execute_mm_encoder` has no request ids. `EncoderCacheManager.check_and_update_cache` is a per-item decision, not a run.
5. **--enable-mm-processor-stats.** Not a `vllm serve` CLI flag (VERIFIED: absent from the observability argparse block; docstring says so). It is an `EngineArgs` field, so it can be set via stage YAML `engine_args`, which is UNVERIFIED end-to-end. Data: `encoder_forward_secs` and `num_encoder_calls` per engine request id, read only via `collective_rpc("get_encoder_timing_stats")`. No HTTP route.

---

## Q1. Process model

### Findings

- **Q1.1 (VERIFIED)** Spawn is forced for stage processes. `prepare_engine_environment()` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and calls `mp.set_start_method("spawn", force=True)` — `vllm_omni/engine/stage_init_utils.py:783-792`. It is called on the stage path (`vllm_omni/engine/stage_runtime.py:734`) and in `vllm_omni/entrypoints/cli/serve.py:1626`.
- **Q1.2 (VERIFIED)** Stage replicas are created with `context = get_mp_context()` and `context.Process(target=StageEngineCoreProc.run_stage_core, ...)` — `vllm_omni/engine/stage_engine_core_proc_manager.py:93`, `:130`. The local-mode alternative (no omni coordinator) uses upstream `CoreEngineProcManager` (`vllm_omni/engine/stage_engine_startup.py:1094-1104`), which also calls `get_mp_context()` and `context.Process` (`vllm/v1/engine/utils.py:261`, `:287`). Either way it is spawn.
- **Q1.3 (VERIFIED)** `get_mp_context()` returns the method from `VLLM_WORKER_MULTIPROC_METHOD` (`vllm/utils/system_utils.py:188-203`). `_maybe_force_spawn` (`:123-170`) can only force spawn further; it does not switch to fork.
- **Q1.4 (VERIFIED)** `StageEngineCoreProc(EngineCoreProc)` is the stage process class (`vllm_omni/engine/stage_engine_core_proc.py:140`). Its entry is `run_stage_core` (`:195`). It holds `self.scheduler` and `self.model_executor`; `_bind_native_data_plane_ready_sink` reads `model_executor.driver_worker` only for `UniProcExecutor` (`:113`).
- **Q1.5 (VERIFIED)** At TP=1 the executor is `UniProcExecutor`. `Executor.get_class` maps `"uni"` to it (`vllm/v1/executor/abstract.py:85-88`). The default backend is `"uni"` when world_size==1 (`vllm/config/parallel.py:1191-1192`). `UniProcExecutor._init_executor` creates `self.driver_worker = WorkerWrapperBase(rpc_rank=0)` in the same process (`vllm/v1/executor/uniproc_executor.py:51-54`). So at TP=1 the scheduler and the worker share one process.
- **Q1.6 (VERIFIED)** At TP>1 the executor is `MultiprocExecutor`, which spawns one worker process per rank with `get_mp_context()` (`vllm/v1/executor/multiproc_executor.py:211`, `:857-870`). The scheduler stays in the stage process; the encoder runs in the worker processes.
- **Q1.7 (VERIFIED)** The EncoderCacheManager is created in the scheduler (`vllm/v1/core/sched/scheduler.py:293`). The worker's encoder output dict is `EncoderCache.encoder_outputs` (`vllm/v1/worker/gpu/mm/encoder_cache.py:10-15`), created in the model runner (`vllm/v1/worker/gpu/model_runner.py:286`).
- **Q1.8 (VERIFIED)** Only the MRv2 runner exists for omni. `OmniModelConfig`/engine args raise `"Model Runner V1 is not supported with vLLM 0.34; select model_runner: v2"` (`vllm_omni/engine/arg_utils.py:484-485`); `use_v2_model_runner` defaults to True (`:346`). The worker model class is `vllm/v1/worker/gpu/model_runner.py` (`GPUModelRunner`, `:206`), selected in `vllm/v1/worker/gpu_worker.py:579-587`.
- **Q1.9 (VERIFIED)** The forkserver branch in `api_server.py:661-666` runs only when `VLLM_WORKER_MULTIPROC_METHOD == "forkserver"`. It is not the default.
- **Q1.10 (VERIFIED)** Child env inheritance: `multiprocessing.util.spawnv_passfds` calls `_posixsubprocess.fork_exec(args, [path], True, passfds, None, None, ...)` — the 5th and 6th arguments are cwd=None and env=None, so the child inherits `os.environ` (`/usr/lib/python3.12/multiprocessing/util.py:450-462`). `spawn.get_command_line` builds `[exe] + opts + ['-c', prog, ...]` (`/usr/lib/python3.12/multiprocessing/spawn.py:83-95`). `opts` come from `subprocess._args_from_interpreter_flags` (`/usr/lib/python3.12/subprocess.py:300-322`), which re-adds `-S` if `no_site` and `-I` if `isolated`.
- **Q1.11 (VERIFIED by scratch run)** Experiment in `/tmp/audit_meas/`: `sitecustomize.py` on `PYTHONPATH`, a spawn child started with `multiprocessing.get_context("spawn")`. Result: the sitecustomize ran in the parent (`argv0=spawn_test.py`) and in the spawned child (`argv0='-c'`). With `python -I`, neither ran. Note: the child also gets `-I` if the parent had it, so `-I` in the serve command kills the hook everywhere.
- **Q1.12 (INFERRED)** So a PYTHONPATH sitecustomize loads in: the API process, the stage-0 `StageEngineCoreProc` (scheduler and, at TP=1, worker), each TP>1 worker, and stage 1 and stage 2 processes. It is not loaded if the parent was started with `-I` or `-S`, or if PYTHONPATH is cleared by a launcher.
- **Q1.13 (VERIFIED, repo)** The existing audit shim already relies on this. `scripts/serve_vllm_omni_cuda.sh:49-54` exports `PYTHONPATH=scripts/audit_vo` and `VO_AUDIT_LOG`. `scripts/audit_vo/sitecustomize.py` patches `vllm.v1.core.encoder_cache_manager.EncoderCacheManager.check_and_update_cache` (engine process) and BAGEL. No `shim.log` exists anywhere in the repo, so an actual in-engine load has **not** been evidenced by a log (UNVERIFIED for the vLLM-Omni serve path).
- **Q1.14 (INFERRED)** The API process is started by `vllm serve` (`vllm_omni/entrypoints/cli/serve.py:203`, `omni_run_server` at `vllm_omni/entrypoints/openai/api_server.py:268`). With the default `api_server_count=1`, there is one API process. With `api_server_count>1`, `run_multi_api_server_omni` (`serve.py:1471`) starts extra API workers — not traced further (UNVERIFIED layout).

### Process list (default single API server, Qwen3-Omni thinker/talker/code2wav)

| Process | Contents | Source |
|---|---|---|
| P0 API server | uvicorn, `AsyncOmni`, orchestrator, stage pools, WS video handler, renderer | `api_server.py:268`; `async_omni_base.py` |
| P1 stage 0 `StageEngineCoreProc_stage0_replica0` (spawn) | EngineCore, `OmniARScheduler` (EncoderCacheManager, KV block pool, prefix hit logic), UniProc worker → MRv2 `GPUModelRunner`, `ModelState`, `EncoderRunner`, vision encoder | `stage_engine_core_proc_manager.py:130`; `uniproc_executor.py:54` |
| P2 stage 1 (talker, spawn) | same structure, talker model | `stage_engine_startup.py:1061-1078` |
| P3 stage 2 (code2wav, spawn) | same structure | same |
| TP>1 extra | one spawned worker per rank (`multiproc_executor.py:857-870`); scheduler stays in P1 | |

**Answer for Q1:** scheduler and encoder run in the same process (P1) at TP=1. A sitecustomize hook loads in P0 and P1 (and in each TP>1 worker), provided the serve command does not use `-I` or `-S`.

---

## Q2. num_cached_tokens path

### Path (stage 0, text output)

1. **Scheduler, first scheduling.** `Request.prefill_stats = PrefillStats()` (`vllm/v1/request.py:228`). At first scheduling (`num_preemptions <= 0`), `prefill_stats.set(num_prompt_tokens, num_local_cached_tokens=num_new_local_computed_tokens, num_external_cached_tokens=...)` (`vllm/v1/core/sched/scheduler.py:1291-1297`). `PrefillStats.num_cached_tokens = local + external` (`vllm/v1/metrics/stats.py:327-340`). The local hit comes from `KVCacheManager.get_computed_blocks` (`vllm/v1/core/kv_cache_manager.py:288-349`).
2. **Scheduler, first emitted output.** The stats are taken and finalized when an output is emitted: upstream `scheduler.py:2510-2513`; omni `vllm_omni/core/sched/omni_ar_scheduler.py:1624-1628` (non-fast path) and `:1425-1427` (fast-decode path). `estimate_cached_tokens` runs before `_free_request` in the stopping case (comment at `omni_ar_scheduler.py:1620-1623`).
3. **EngineCoreOutput.** `EngineCoreOutput.prefill_stats` (`vllm/v1/engine/__init__.py:286`). Omni passes it through `_append_request_output(prefill_stats=...)` (`vllm_omni/core/sched/omni_scheduler_mixin.py:1056`, `:1089`).
4. **Output processor.** Upstream text path sets `req_state.num_cached_tokens` only while `is_prefilling` (`vllm/v1/engine/output_processor.py:793-797`), then `is_prefilling=False` (`:803`). Default is 0 (`:232`). Omni mm-only path sets it unconditionally (`vllm_omni/outputs/output_processor.py:887-888`) and copies it into the output at `:1022`. Text path builds `RequestOutput(num_cached_tokens=req_state.num_cached_tokens)` (`vllm/v1/engine/output_processor.py:1044`).
5. **OmniRequestOutput.** `num_cached_tokens` is a field (`vllm_omni/outputs/__init__.py:197`) and is in the copy list (`:132`; copy at `:228-251`).
6. **Orchestrator to client.** Only outputs from final-output stages reach the client: `_compute_final_output_stage_ids` picks stages with `final_output` whose type is among requested modalities (`vllm_omni/entrypoints/omni_base.py:476-485`); client completion requires `pool.final_output` and `final_output_stage_ids` (`vllm_omni/engine/orchestrator.py:2376-2380`).

### Per-case answers

- **(a) MRv2 runner: VERIFIED populated.** It is the only runner for omni (Q1.8). The MRv2 runner returns `ModelRunnerOutput`; the scheduler does the `prefill_stats` work above.
- **(b) V1 runner: N/A.** Refused for omni in vLLM 0.34 (`vllm_omni/engine/arg_utils.py:484-485`). There is no `vllm/v1/worker/gpu_model_runner.py` in the installed vLLM (`vllm/v1/worker/gpu_worker.py:579-587` builds only the `gpu/model_runner.py` class, or the MM-encoder-only runner).
- **(c) async_chunk stage 0: INFERRED, with a real gap.** The same omni AR scheduler is used (`vllm_omni/config/stage_config.py:372-378`; `OmniARAsyncScheduler` for async scheduling). The stats are taken at the first step that has tokens (`omni_ar_scheduler.py:1624-1628`), but emission is gated by `_should_emit_engine_output` (`:100-112`): with `use_v2_model_runner and async_chunk and not final_output`, `emit_every_step` is False. The gate at `:1776-1779` then does not emit a non-stopping step. So the stats are taken and dropped. The output then reads `num_cached_tokens = 0` (default, `vllm/v1/engine/output_processor.py:232`), which looks like a miss. For a max_tokens=1 warmup the step also stops, so it is emitted. This affects multi-token requests on a non-final stage 0. Whether the Qwen3-Omni stage-0 config sets `final_output` is UNVERIFIED.
- **(d) max_tokens=1: VERIFIED populated.** The first step with tokens also stops (`finish_reason=length`); the stop branch emits `prefill_stats` (`omni_ar_scheduler.py:1624-1628`, `:1776-1792` with `stopped=True`) and the stats are taken before `_free_request`.

### Consumer loops

- **`_drive_warmup_generate` drops everything.** `async def _drive(): async for _ in result_gen: pass` (`vllm_omni/entrypoints/openai/video_stream_base.py:1363-1365`). A warmup cancelled before its first token emits no output, so no `num_cached_tokens` is ever seen for it.
- **`_drive_query` reads `output`.** `async for output in result_gen` (`video_stream_base.py:1011`), with `isinstance(output, OmniRequestOutput)` (`:1014-1015`). `num_cached_tokens` is cumulative on each text output after prefill (req_state field persists; `output_processor.py:1044`).
- **Audio modality.** If stage 0 is not in `final_output_stage_ids` (audio-only), the client sees only stage-2 outputs; stage-0 `num_cached_tokens` is not visible (INFERRED from `omni_base.py:476-485`, `orchestrator.py:2376-2380`).
- **WS protocol.** No usage or cached-token field is sent: the protocol list is `video.frame.ack`, `video.frames.consumed`, `response.start`, `response.text.delta/done`, `response.output_audio.*`, `session.done`, `error` (`video_stream_base.py:18-27`). The only timing log is `[TIMING] mode=... total=... first_text=... first_audio=... audio_chunks=...` (`video_stream_base.py:1128-1136`).

### Least-change options

1. **Log-facing, query only.** In the `_drive_query` loop (`video_stream_base.py:1011`), when `out_type == "text"` (or stage is final text), record `getattr(output, "num_cached_tokens", None)` and add it to the existing `[TIMING]` line (`:1128`). Log only; no protocol change.
2. **Client-facing, query only.** Add the same value as an additive key in `response.text.done` (`:1054`, `:1094`). The protocol is JSON, so an extra key is additive.
3. **Warmup, log-facing.** Replace the `pass` at `:1364` with recording the last output's `num_cached_tokens`. This only covers warmups that reach their first token.
4. **Warmup, reliable.** Engine-side (sitecustomize): wrap `KVCacheManager.get_computed_blocks` (`vllm/v1/core/kv_cache_manager.py:288`), which returns `(blocks, num_new_computed_tokens, shared_prefix_boundary)`, called at admission from `scheduler.py:684-691` and `:1246`. Or read `request.prefill_stats` at `scheduler.py:1291-1297`. This gives the hit count for every admitted request, including aborted ones.

Note: `max_cache_hit_length` is `prompt_len - 1` (`kv_cache_manager.py:311-313`), so a fully cached prompt still recomputes the last token.

---

## Q3. Abort semantics for warmup requests

### Abort path (VERIFIED)

- `AsyncOmni.abort` settles pending submits then calls the orchestrator (`vllm_omni/entrypoints/async_omni.py:920-930`).
- `_cancel_warmup` cancels the warmup task, then calls `engine_client.abort(request_id)` (`video_stream_base.py:521-533`, abort at `:531`).
- Orchestrator `_handle_abort` (`vllm_omni/engine/orchestrator.py:889`) calls `_abort_request_ids`, which forwards to all stage pools (`:929`).
- Engine core: `EngineCore.abort_requests` → `scheduler.finish_requests(request_ids, FINISHED_ABORTED)` (`vllm/v1/engine/core.py:610-615`).
- Omni scheduler: `OmniARScheduler.finish_requests` (`vllm_omni/core/sched/omni_ar_scheduler.py:381-402`) calls super; `_free_request` (`:2490-2560`) releases encoder cache and calls the upstream free path.
- Upstream `Scheduler.finish_requests` (`vllm/v1/core/sched/scheduler.py:3100-3170`) removes the request from queues and calls `_free_request` (`:3170-3202`), which calls `encoder_cache_manager.free(request)` (`:3193` region) then `_free_blocks` (`:3204`) → `_free_request_blocks` (`:3224-3232`) → `kv_cache_manager.free` (`vllm/v1/core/kv_cache_manager.py:650-659`) → `coordinator.free`.
- Block pool: `free_blocks` (`vllm/v1/core/block_pool.py:798-835`). Ref count decremented. Blocks reaching ref 0 that have a hash (`block_hash is not None`, caching on) go to `blocks_to_evict_last` and are **appended to the tail** of the free queue (LRU). Blocks without a hash are prepended (evicted first).
- Eviction on reuse: `get_new_blocks` pops from the head and calls `_maybe_evict_cached_block` (`block_pool.py:681-713`, `:751-772`). A cached block keeps its hash until it is popped for reuse.

### Which blocks survive (VERIFIED for mechanism, INFERRED for the warmup case)

- **Full blocks are hashed at scheduling time, not at completion.** `KVCacheManager.allocate_slots` calls `coordinator.cache_blocks(request, num_tokens_to_cache)` with `num_tokens_to_cache = min(total_computed + num_new_tokens, request.num_tokens)` (`kv_cache_manager.py:634-646`). `SingleTypeKVCacheManager.cache_blocks` caches `num_full_blocks = min(num_tokens, num_hashed_tokens) // block_size` (`vllm/v1/core/single_type_kv_cache_manager.py:482-545`, `:508`).
- **So:** every full block of the warmup prompt that was scheduled before the abort is in the hash map after the abort, and is reusable by the query until evicted. The query can hit it only if the token sequence and multimodal hashes match from block 0 through that block (hash chain). The warmup and query share the system + history + frames prefix (see prior audit); blocks past the point where the warmup's trailing generation prompt diverges from the query's user text are not reusable (INFERRED from prompt construction, `video_stream_base.py:1342-1412`).
- **Freed-while-in-flight.** `defer_block_free` is True only for `kv_consumer` with multiple in-flight batches (`scheduler.py:190-193`). Otherwise blocks are freed at once, and the query's step is ordered after the warmup's in-flight step on the same stream (INFERRED safe).

### Partial last block

- **Not reusable by default (VERIFIED).** Only full blocks are cached (`single_type_kv_cache_manager.py:508`). Partial tail caching exists only in `FullAttentionManager.cache_blocks` when `block_size != hash_block_size` (`single_type_kv_cache_manager.py:843-862`). `hash_block_size` is `prefix_match_unit` or the GCD of block sizes (`vllm/v1/core/kv_cache_utils.py:880-887`). Omni refuses sub-block hits (`vllm_omni/core/prefix_cache/group_view.py:213-227`). Nothing in vllm_omni sets `prefix_match_unit`.

### Omni-specific paths

- **KV release:** only a Mooncake-specific `_free_request_blocks` override (`omni_ar_scheduler.py:321-336`, active only with `MooncakeConnector` and caching off). No other omni release of KV blocks.
- **Omni prefix cache (`vllm_omni/core/prefix_cache/`, MRv2, Qwen3-Omni opt-in at `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py:297`):** on abort, pending writes for the aborted request are forced onto the high-priority copy queue so "a block hash that entered the batch must land in the cache, abort included" (`vllm_omni/core/prefix_cache/manager.py:476-500`). So the omni hidden-state/mm rows for hashed blocks also survive abort. Enablement follows `enable_prefix_caching` (upstream default True, `vllm/config/cache.py:267`; Qwen3-Omni deploy value UNVERIFIED here, see 11-25 audit).
- **Double abort:** `AsyncOmni.generate` also aborts on `CancelledError`/`GeneratorExit` (`vllm_omni/entrypoints/async_omni.py:403-408`). Harmless: `finish_requests` skips unknown or finished ids (`scheduler.py:3130-3132`). It also records a `client_disconnect` failure for each warmup (cosmetic for metrics).

### Answer for Q3

- Full blocks hashed before the abort: **yes**, reusable by the query until evicted (LRU tail, popped on reuse).
- Condition: exact token and mm-hash match from block 0 through the block. Blocks past the warmup's trailing-prompt divergence do not match.
- Partially computed last block: **no** by default.
- Warmup cancelled before its first token: its blocks are still hashed for whatever was scheduled, but nothing is emitted to the client; use the admission-time hit count (Q2 option 4).

---

## Q4. Engine-side encoder timing hooks

### Candidate table

| Hook | Class.method (file:line) | Granularity | Request id? | Process | Notes |
|---|---|---|---|---|---|
| (i) V1 runner | n/a in 0.34 for omni | — | — | — | Refused (`arg_utils.py:484-485`) |
| (ii) MRv2 batch executor | `vllm.v1.worker.gpu.mm.encoder_runner.EncoderRunner.execute_mm_encoder(self, mm_kwargs)` (`encoder_runner.py:150-172`) | per modality batch (`group_and_batch_mm_kwargs`), items sorted by modality | **No.** Only `(modality, item)` pairs | Worker (= stage process at TP=1) | Also used by profiling (`:139`) |
| (ii') MRv2 timing wrapper | `EncoderRunner.timed_encoder_operation(self, request_ids)` (`encoder_runner.py:178-196`) | per model-state call, wall time split equally over request ids | Yes (dict keys passed in) | Worker | Only active when `enable_timing` (`:179-181`); called only if `mm_kwargs` non-empty (`model_states/interface.py:199-206`) |
| (iii) MRv2 per-step chokepoint | `vllm.v1.worker.gpu.model_states.interface.ModelState.execute_mm_encoder(self, scheduled_encoder_inputs)` (`interface.py:191-207`) | per step; dict `{engine_req_id: [input_idx,...]}` | **Yes** (dict keys) | Worker (= stage process at TP=1) | Called by `DefaultModelState.prepare_inputs_embeds` (`model_states/default.py:108`) and `OmniModelState.prepare_inputs_embeds` (`vllm_omni/worker_v2/model_states/omni_model_state.py:449-450`). No override in either (only BAGEL overrides, `vllm_omni/model_executor/models/bagel/model_state.py:26`) |
| (iv) Scheduler decision | `vllm.v1.core.encoder_cache_manager.EncoderCacheManager.check_and_update_cache(self, request, input_id)` (`encoder_cache_manager.py:111-139`) | per item, per scheduling attempt | Yes (`request`) | Engine-core = scheduler process (P1) | Returns True only if the item is cached. False does not mean a run: `can_allocate` may still refuse. Called at `scheduler.py:2198` |

Also relevant: `_try_schedule_encoder_inputs` (`scheduler.py:2064-2200`) only considers items whose span overlaps `[num_computed_tokens, num_computed_tokens + num_new_tokens)` (`lo, hi = get_mm_features_in_window(..., start=num_computed_tokens, ...)`, `:2109`; `get_mm_features_in_window` excludes features ending at or before `start`, `vllm/multimodal/utils.py:122-142`). Frames fully covered by a prefix hit therefore produce **no** ECM check and **no** encoder run. The existing shim's `ecm_check` log will not show them.

Scheduler output: `scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule` (`scheduler.py:1045`, `:1665`). These keys are engine request ids.

### Process and id details

- **Process.** (ii), (ii'), (iii) run in the worker. At TP=1 that is the stage-0 `StageEngineCoreProc` process (`uniproc_executor.py:54`). (iv) runs in the scheduler, same process at TP=1.
- **Attribution.** Stage requests carry the orchestrator id unchanged (`vllm_omni/engine/orchestrator.py:234-235`; `build_engine_core_request_from_tokens`). The orchestrator id is `"<external>-<8 chars>"` (`vllm_omni/entrypoints/async_omni_base.py:308-318`; `async_omni.py:211-218`). vLLM's `InputProcessor.assign_request_id` (`vllm/v1/engine/input_processor.py:342-357`) is not on the omni stage path (only `async_llm.py:535`, `:644`). So an engine id should start with the warmup's external id `video-warmup-<12 hex>-` (INFERRED; the existing shim already logs `request.request_id[:24]`).
- **Timing semantics.** `timed_encoder_operation` synchronizes before and after (`encoder_runner.py:183-196`) and divides the batch's wall time equally among its request ids. With one session's warmup, the division is exact; with several requests in one step it is an approximation.
- **Skipped items.** `prepare_mm_inputs` drops items already in `encoder_cache.encoder_outputs` (`encoder_runner.py:81-~100`). So `ModelState.execute_mm_encoder`'s dict can contain items that will not run.

### Recommendation

**Single best hook: `vllm.v1.worker.gpu.model_states.interface.ModelState.execute_mm_encoder`** (wrap the class method, so both `DefaultModelState` and `OmniModelState` pick it up).

Why: it is the only point that has (1) engine request ids, (2) the exact items the scheduler chose to encode in that step (cache hits and prefix-covered frames already removed by the scheduler), and (3) the worker-side execution, so GPU time can be synchronized. Inside the wrapper:

- Pre-filter the dict the same way `prepare_mm_inputs` does (skip items whose `identifier` is already in `self.encoder_runner.encoder_cache.encoder_outputs`), so only real runs are counted.
- Record per request id: number of items, modality, `mm_position.length`, and `time` with `torch.accelerator.synchronize()` before and after (or CUDA events).
- If multiple requests share a step, divide the time as `timed_encoder_operation` does, and say so in the log.

Alternative with the same information: patch `EncoderRunner.timed_encoder_operation` directly, bypassing its `enable_timing` gate. It is entered once per real encoder run with request ids (`interface.py:202-206`). It attributes to all requests in the step, not only the ones with real items.

Companion engine-side hook for cache hit vs prefix hit: `EncoderCacheManager.check_and_update_cache` (`encoder_cache_manager.py:111`), which is the existing shim's hook.

---

## Q5. --enable-mm-processor-stats

### Findings

- **CLI wiring: not on `vllm serve`.** `enable_mm_processor_stats` is an `EngineArgs` field (`vllm/engine/arg_utils.py:771`) passed into `ObservabilityConfig` (`:2262`). The observability CLI block (`:1696-1750`) does not add it to argparse. The `ObservabilityConfig` docstring says "for internal use only (e.g., benchmarks) and is not exposed as a CLI argument" (`vllm/config/observability.py:96-99`). VERIFIED by static reading. Runtime argparse rejection is INFERRED (not run).
- **The only CLI-side user is the benchmark.** `vllm/benchmarks/mm_processor.py:369-374` does `EngineArgs.add_cli_args(parser)` then `parser.set_defaults(enable_mm_processor_stats=True)`. vllm-omni has no reference to the flag (grep: none under `vllm_omni/`).
- **Omni engine args accept it.** `OmniEngineArgs(EngineArgs)` (`vllm_omni/engine/arg_utils.py:282`) inherits the field. Stage YAML `engine_args` are filtered by `filter_dataclass_kwargs(OmniEngineArgs, ...)` (`vllm_omni/engine/stage_init_utils.py:1596`), which silently drops unknown keys. So `enable_mm_processor_stats: true` in a stage's `engine_args` should reach `ObservabilityConfig` (INFERRED; the YAML route is UNVERIFIED end-to-end). Unknown keys are dropped silently, so a typo would fail quietly.
- **What it records.** When enabled, `ModelState` sets `EncoderRunner.enable_timing` (`vllm/v1/worker/gpu/model_states/interface.py:84-95`). `timed_encoder_operation` accumulates `EncoderTimingStats(encoder_forward_secs, num_encoder_calls)` per request id (`vllm/v1/worker/gpu/mm/encoder_runner.py:178-196`; stats class `vllm/v1/worker/utils.py:1007-1020`). `get_encoder_timing_stats` returns and clears the registry (`encoder_runner.py:198-206`).
- **Exposure.** Only through worker RPC: `vllm/v1/worker/gpu_worker.py:1323-1325` and `vllm/v1/worker/gpu/model_runner.py:628-630`, consumed by `collective_rpc("get_encoder_timing_stats")` in `vllm/benchmarks/mm_processor.py:82`. No HTTP route in vLLM or vLLM-Omni reads it (VERIFIED by grep). Access from the omni API process would need an RPC to the stage (omni has `_engine_core_rpc(...)` helpers, e.g. `async_omni.py:937-939`) or an in-process shim.
- **Also in the flag's scope.** `MultiModalTimingRegistry` (`vllm/multimodal/registry.py:165-185`) records frontend preprocessing timings, gated by the same flag. Not an encoder timing.
- **Attributability of warmup ids.** VERIFIED that the key is the engine-core request id, which per the Q4 id analysis should carry the warmup's external prefix (INFERRED). Timing is per request (not per frame): one warmup's record covers all encoder runs it triggered. A second warmup or query in the same window is separate only if it ran in a separate step or the stats were drained between them.

### Answer for Q5

Not a `vllm serve` CLI flag. It is usable through the stage YAML `engine_args` (INFERRED, UNVERIFIED end-to-end) or a shim. Data: encoder forward seconds and call count per engine request id, drained by worker RPC; no HTTP exposure. Warmup ids are attributable by prefix (INFERRED). The Q4 hook (`ModelState.execute_mm_encoder`) gives the same information with a timestamp per run and no dependence on the flag or on RPC draining.

---

## Recommended hook list (exact names)

1. **Encoder run per request (primary):** wrap `vllm.v1.worker.gpu.model_states.interface.ModelState.execute_mm_encoder` (method def `interface.py:191`). Pre-filter `scheduled_encoder_inputs` against `self.encoder_runner.encoder_cache.encoder_outputs`. Time with `torch.accelerator.synchronize()`. Log engine request id prefix, items, modality, length. Runs in the stage process (TP=1).
2. **Cache-hit decision (secondary, engine process):** wrap `vllm.v1.core.encoder_cache_manager.EncoderCacheManager.check_and_update_cache` (`encoder_cache_manager.py:111`). Already done by `scripts/audit_vo/sitecustomize.py`. It does not see prefix-covered frames.
3. **Prefix hit tokens per request (engine process, reliable for aborted warmups):** wrap `vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks` (`kv_cache_manager.py:288`), or read `Request.prefill_stats` at `vllm/v1/core/sched/scheduler.py:1291-1297`. Log `num_new_computed_tokens` with the request id.
4. **Warmup and query outputs (API process, log-only):** in `vllm_omni/entrypoints/openai/video_stream_base.py`, record `num_cached_tokens` at `_drive` (`:1364`) and in the `_drive_query` loop (`:1011`), and add to the `[TIMING]` log (`:1128`). Treat `0` as ambiguous (see Q2 caveat).
5. **Warmup start/finish timing:** not reported by the handler today. Add wall-clock stamps around `_drive_warmup_generate` (`video_stream_base.py:1342`) and `_cancel_warmup` (`:521`), in the API process.

Measurement design note: for a "cache hit" claim, use (3) (prefix tokens actually reused by the query), not `num_cached_tokens` alone, and for an "encoder run" claim use (1). Check that the sitecustomize loads in the stage process before trusting either: a log line from the hook is the only evidence (no such log exists in the repo yet).

---

## Could not verify

- **Runtime behaviour.** Nothing was run beyond the scratch spawn experiment (no vLLM import, no GPU, no server). All vLLM-side claims are code-read.
- **`--enable-mm-processor-stats` on the CLI.** Static reading only; whether argparse rejects it on `vllm serve --omni` is INFERRED.
- **YAML route for `enable_mm_processor_stats`.** Filter and dataclass inheritance are read; end-to-end acceptance by the Qwen3-Omni stage YAML is UNVERIFIED.
- **Stage-0 `final_output` and async_chunk flags for the Qwen3-Omni deploy config.** These decide whether the Q2(c) drop case applies, and whether stage-0 text reaches the client in audio sessions. Not checked; see `vllm_omni/deploy/` and `vllm_omni/config/stage_config.py`.
- **Engine request id format in the stage process.** INFERRED from `orchestrator.py:234-235` and `async_omni_base.py:308-318`; not observed in a log.
- **Multiple API servers (`api_server_count>1`) layout.** Not traced (`serve.py:1471`).
- **Block hash extra keys for multimodal items.** Assumed to include mm hashes for the shared frames; not traced in `vllm/v1/core/kv_cache_utils.py`. The hit-chain condition in Q3 depends on this.
- **Whether `_stop_after_submit` cancels the drive task** (so the generator-cancel abort in `AsyncOmni.generate` fires in addition to `_cancel_warmup`'s abort). Not traced past `video_stream_base.py:1319`.
- **Omni prefix cache's effect on encoder skipping.** Encoder skipping was traced only at the scheduler level (`scheduler.py:2109`). Whether the omni prefix cache also restores mm rows without an encoder run was not traced in `vllm_omni/core/prefix_cache/manager.py`.
- **Qwen3-Omni enable_prefix_caching value** in the deploy config used by the study (default True upstream, `vllm/config/cache.py:267`).
- **Warmup trailing-prompt divergence.** Which exact blocks diverge from the query was inferred from prompt structure, not computed from tokenization.
- **TP>1 worker process details** beyond `multiproc_executor.py:857-870`.
