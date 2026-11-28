# M* measurement hooks for Qwen3-Omni chunk timing (metrics, step timing, arrival, benchmarks, wrapper plan)

## Question asked

Read-only source audit of M* at commit 2c8ffde8f, for a study that must timestamp, per request:
(i) when each audio/video chunk arrives at the server, (ii) when its vision/audio encoder completes,
(iii) when Thinker prefill for it completes, and (iv) when a query arrives and its first token is produced.

- Q1. Full metric catalog (names, type, labels, buckets, granularity); any request id in labels or exemplars; per-request timeline or trace; `[TIMING]` or per-request timing in OpenAI adapter responses.
- Q2. How `mstar_step_seconds` is recorded (clock, device sync); whether the step batch exposes request ids and node name; whether per-request encoder time is readable without patching.
- Q3. Arrival timestamps in streaming/realtime paths; clock shared with encoder timing; the nearest existing mechanism if Qwen3-Omni has no streaming input.
- Q4. Existing benchmark and test scripts that measure latency or timing, and whether they can be reused.
- Q5. Minimal wrapper-only hook plan: targets, request id source, what to log, overhead, a clock for cross-process alignment, and whether engine workers inherit PYTHONPATH `sitecustomize`.

## Scope

- Repo: `/home/chaofan/workspaces/mm_reuse_study/external/mstar`, commit `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` (`git rev-parse HEAD`). Package `mstar/`, plus `docs/mintlify/` (metrics, serving), `benchmark/`, `perf_testing/`, `test/`.
- Read: `mstar/observability/{catalog,metrics,recorders}.py`, `mstar/worker/{worker_metrics,step_pipeline,step_executor,executing_batch,scheduled_batch,phase_timer,input_ingest,worker}.py`, `mstar/engine/engine.py` (exec, ExecutingBatch, _skip_cached_prefix), `mstar/engine/resources/{step,runner,base}.py`, `mstar/engine/resources/kv/manager.py`, `mstar/profile/{worker,format,display}.py`, `mstar/api_server/{entrypoint,data_worker,request_types,openai/serving_chat,openai/serving_realtime,openai/router}.py`, `mstar/conductor/{conductor,request_info,mps}.py`, `mstar/cli/main.py`, `mstar/platforms/__init__.py`, `mstar/streaming/topology.py`, `mstar/model/qwen3_omni/qwen3_omni_model.py`, `mstar/utils/profiler.py`, `benchmark/`, `test/qwen3-omni/`.
- Prior notes used only as cross-checks: `agents_results/2026-11-25_qwen3-omni-g0-code-audit.md` (its Qwen3-Omni prefill and prefix-cache claims were re-checked where cited here), `agents_results/2026-11-26_measurement-hooks-process-cache-timing.md` (vLLM-Omni, not M*).
- No server started, no GPU used (`CUDA_VISIBLE_DEVICES=` for the one Python probe), no file under `external/` modified. Scratch only: `/tmp/audit_mstar_meas/hookprobe/` (a `sitecustomize.py` marker and a spawn probe, CPU only, does not import mstar or torch).

Line numbers are for the commit above.

---

## Findings

### Q1. Metric catalog, request ids, timelines, adapter timing

**Verdict.** Every metric is an aggregate. No metric has a request-id label or an exemplar. Per-request timing exists only in the in-memory `RequestProfile`, which is printed as text with `--log-stats`. No trace file or event log exists. OpenAI responses carry no timing fields.

Catalog (VERIFIED, `mstar/observability/catalog.py`; API-server families in `metrics.py`). Prometheus exports counters with a `_total` suffix.

| Family | Type | Labels | Buckets | Granularity | Source |
|---|---|---|---|---|---|
| `mstar_step_seconds` | histogram | component (node), graph_walk | 0.5 ms .. 30 s | per engine step, node-level | catalog.py:87-97 |
| `mstar_batch_size` | histogram | component, graph_walk | 1 .. 1024 rows | per engine step | catalog.py:98-103 |
| `mstar_component_pending_steps` | gauge | component | - | node-level, sampled | catalog.py:155-160 |
| `mstar_component_in_flight` | gauge | component | - | node-level (conductor status) | catalog.py:41-46 |
| `mstar_prefix_cache_query_tokens_total` | counter | component | - | per node, aggregate (counted once per request per stream) | catalog.py:104-108; worker_metrics.py:114,189-191; kv/manager.py:1051-1055 |
| `mstar_prefix_cache_hit_tokens_total` | counter | component | - | per node, aggregate | catalog.py:109-113 |
| `mstar_kv_pages_used/total/cached/lent` | gauge | component, resource | - | per node | catalog.py:65-86 |
| `mstar_kv_preemptions_total`, `_drops_`, `_rebuild_tokens_`, `_rebuild_cached_pages_`, `_preempted_again_`, `_returns_preempted_`, `_take_backs_`, `_take_back_waits_` | counter | component | - | per node | catalog.py:114-154 |
| `mstar_rebuild_host_bytes_used/total` | gauge | component | - | per node | catalog.py:162-173 |
| `mstar_transfer_bytes_total`, `mstar_transfers_total` | counter | src, dst, transport | - | per edge | catalog.py:184-193 |
| `mstar_transfer_seconds` | histogram | src, dst, transport | 10 us .. 10 s | per edge read | catalog.py:194-200 |
| `mstar_routing_predicted_saving_total` | counter | term | - | conductor, aggregate | catalog.py:47-53 |
| `mstar_conductor_requests` | gauge | state | - | aggregate | catalog.py:36-40 |
| `mstar_requests_total` (`mstar_requests` in code) | counter | model_name, output_modalities, outcome | - | aggregate | metrics.py:374 |
| `mstar_e2e_request_latency_seconds` | histogram | model_name, output_modalities | E2E_BUCKETS (0.05 .. 1200 s) | per request, aggregated, no id | metrics.py:380; buckets 77-80; observed 521 |
| `mstar_time_to_first_output_seconds` | histogram | model_name, output_modalities | FIRST_OUTPUT_BUCKETS (10 ms .. 600 s) | per request, aggregated | metrics.py:387; observed 523 |
| `mstar_time_to_first_chunk_seconds` | histogram | model_name, modality | FIRST_OUTPUT_BUCKETS | per request and modality, aggregated | metrics.py:449; observed 535-540 |
| `mstar_request_queue_seconds` | histogram | model_name, output_modalities | 1 ms .. 600 s | per request (API-only family) | catalog.py:54-63; metrics.py:396; observed 533-534 |
| `mstar_request_preprocess_seconds`, `_engine_seconds`, `_delivery_seconds` | histogram | model_name, output_modalities | STAGE / E2E buckets | per request, aggregated | metrics.py:415-430; observed 524-530 |
| `mstar_data_worker_preprocess_seconds` (+ `_failures`) | histogram (+ counter) | model_name, input_modalities | STAGE buckets | per request, aggregated | metrics.py:458-469; observed 578 |
| `mstar_upload_size_bytes` | histogram | model_name, modality | 1 KiB .. 1 GiB | per input file, no timestamp | metrics.py:470; observed 579-585 |
| `mstar_output_chunks_total`, `mstar_output_bytes_total`, `mstar_output_audio_seconds_total` | counter | model_name, modality | - | per output chunk, aggregate | metrics.py:434-448; record_chunk 542-558 |
| `mstar_http_refusals_total` | counter | model_name, route, status | - | per refused HTTP request | metrics.py:249 |
| gauges: `mstar_requests_in_flight`, `mstar_data_worker_queue_depth`, `mstar_conductor_status_age_seconds` | gauge | model_name | - | scraped | metrics.py:400, 476, 191 |

Request ids: none. The module docstring says every label takes values from a fixed set (metrics.py:1-9). No `exemplar` string appears anywhere under `mstar/` (grep).

Per-request timeline (VERIFIED):
- `RequestProfile` (`mstar/profile/format.py:97-104`) holds `RequestTiming` (recv, preprocess-done, conductor ingest, first chunk, last chunk, conductor done, finish; `format.py:80-95`) and `graph_timings: list[GraphTiming]` (per node and graph walk: `exec_count`, `total_time`, `forward_time`, `preprocess_time`, `postprocess_time`, `first_start`, `last_end`; `format.py:48-78`).
- It is created per request at `entrypoint.py:940-942` (`RequestTiming(recv_time=time.perf_counter())`) and printed by `_finalize_profile` (`entrypoint.py:1917-1934`) only when `--log-stats` is set (`entrypoint.py:1925`). The printed text (`mstar/profile/display.py`) shows the timeline and per-node totals and counts, not absolute timestamps.
- Gating: `--log-stats` sets `enable_prof` in the data worker, conductor and workers (`entrypoint.py:624`; `conductor.py` spawn args; `worker.py:207,238`).
- No event log, trace file, OpenTelemetry or Chrome-trace output. NVTX ranges exist only under `--enable-nvtx` (`mstar/utils/profiler.py:61-67`). `MSTAR_PHASE_TIMING` writes per-window percentiles, not per-request samples (`mstar/worker/phase_timer.py:145-170`).

OpenAI adapter responses (VERIFIED): no timing field. The `[TIMING]` string does not occur in `mstar/`. The chat usage object carries only `prompt_tokens`, `completion_tokens`, `total_tokens` (`mstar/api_server/openai/serving_chat.py:79-83`). No cached-token field exists in the API or conductor code (grep for `cached_tokens` and `prompt_tokens_details` is empty). The docs say prefix counters are per node, aggregate (`docs/mintlify/serving.mdx:492`).

### Q2. Step and node timing; per-request attribution

**How `mstar_step_seconds` is recorded (VERIFIED).**
- Recorded by `WorkerMetrics.step_done` (`mstar/worker/worker_metrics.py:135-151`), called from `StepPipeline._finish_step` (`mstar/worker/step_pipeline.py:658-679`, the call at 679). `_finish_step` is reached from `_finish_chain_step` (`step_pipeline.py:346`, call at 411) for the chain and from `_finish_side` (`step_pipeline.py:579`, call at 597) for side lanes.
- Clock: `time.perf_counter()` (worker_metrics.py:143). Start = `max(submitted_at, last_done)`, where `submitted_at` is stamped in `StepExecutor.submit` (`step_executor.py:658-676`, line 670) only when metrics are on and the step is not speculated. Start is therefore the GPU-thread hand-off, or the end of the previous step if later (pipelined steps are not double-counted; worker_metrics.py:143-144).
- End: after `executor.postprocess` returns (step_pipeline.py:658-679).
- Sync: `postprocess` waits for the step's completion event before postprocessing (`step_executor.py:858-861`; `wait_for_step` at 691-700 uses `completion_event.synchronize()`, or `torch.accelerator.synchronize(device)` if the step has no completion event, 697-700). So the step end is after that step's GPU work on its execution stream. The metric adds no sync of its own.
- Gating: nodes this rank leads only (`worker_metrics.py:12-15`, 148); steps with no forward are skipped (148); counting starts at `ENABLE_METRICS`, sent after warmups (`input_ingest.py:106`, `entrypoint.py:798-802`).
- Labels are (node, graph_walk) only. Request ids are not recorded.

**What the step batch exposes (VERIFIED).**
- `ExecutingBatch` (`mstar/engine/engine.py:435`): `node_name` (436), `per_request_info: Mapping[int, CurrentForwardPassInfo]` (442), `step_context: StepContext` (443), `exec_timings: ExecTimings` (561). `request_ids` is a property over `step_context.request_ids` (engine.py:574-575).
- `StepContext` (`mstar/engine/resources/step.py:108-115`): `request_ids` (worker integer handles, not strings), `graph_walk`, `slot`, `capture`.
- `CurrentForwardPassInfo` (`mstar/conductor/request_info.py:16-33`): `request_id: str` is the wire identity; `rid_handle` is the worker-local handle. Keys of `per_request_info` are handles (engine.py:438-440; `GraphRuntime.get_rid_handle` / `get_rid_string`, `mstar/graph/runtime/base.py:~934-960`).
- CUDA-graph padding adds rows with negative handles (engine.py:438-440). Filter them.
- The wire id is `"<client_request_id>#<generation>"`: `entrypoint.py:933` (`key = f"{request_id}#{next(self._generation)}"`), mapped back at 935 (`_client_ids[key] = request_id`). The data worker sends `request_id=input.request_id` (the key) to the conductor (`data_worker.py:1124-1126`); the client id goes separately as `client_request_id`. Splitting on the last `#` recovers the client id.
- `Engine.exec` (`engine.py:1893`) is the forward; `exec_and_postprocess` calls it (`engine.py:2606-2615`). `exec_timings.start` is stamped at exec entry when profiling (`engine.py:1904-1905`); `fwd_start` before launch (2453-2454).
- Per-request graph timing is `WorkerProfileInfo.register_end` (`mstar/profile/worker.py:63-100`), called from `StepExecutor._send_input` (`step_executor.py:1287`), inside `postprocess` after the wait at 861. Every row of a batch receives the batch's start and end (worker.py:63-100): batched steps are attributed to the whole batch window.
- The fwd end is stamped after the completion-event wait (`step_executor.py:861-864`). The docs say `fwd` is "not synced" (`docs/mintlify/serving.mdx:451`). The code contradicts the docs: `fwd` includes the GPU completion wait. Note this in any use of log-stats `fwd`.

**Can per-request encoder time be read without patching? (VERIFIED, with limits.)** Yes, partially. Run with `--log-stats-file FILE`. Each request's profile lists, for each `(node, graph_walk)`, the total over the request, the average per execution, and `n`. Node names: `vision_encoder` and `audio_encoder` on walks `prefill_vision` and `prefill_audio`; `Thinker` on `prefill_vision`, `prefill_text`, `thinker_decode` (`mstar/model/qwen3_omni/qwen3_omni_model.py:530-601`, 746-748, 765-767). `benchmark/request_timeline/report.py` already parses this file. Limits:
- Totals and counts, not absolute timestamps. The first and last timestamps are in the object (`first_start`, `last_end`, format.py:48-78) but are not printed.
- Batched steps carry the batch window for every row (worker.py:63-100).
- Thinker prefill for a video is one `prefill_vision` walk per attachment, chunked at 1024 tokens (`qwen3_omni_model.py:208-221` and 480-493; the 11-25 audit). So a per-request Thinker prefill total spans several steps.

Absolute encoder start and end per request therefore need a wrapper (Q5).

### Q3. Arrival timestamps and streaming paths

**No streaming input exists for Qwen3-Omni (VERIFIED).**
- Its partition topology is Thinker, Talker, Code2Wav with three internal connections and no `CLIENT` source (`qwen3_omni_model.py:780-803`). No `ClientAudio`/`ClientVideo` stream is declared (grep of `qwen3_omni` for `ClientAudio`, `client_connections`, `CLIENT` is empty).
- `takes_stream_input = bool(self.client_streams)` is False (`entrypoint.py:599-601`). An append to a stream the model does not declare is refused (`entrypoint.py:1093-1097`, "this model takes ... none").
- So a Qwen3-Omni "chunk" is an attachment inside a whole request. On `/generate/ws`, each frame is one message (`entrypoint.py:2712-2720`); for this model an append frame that names a stream is refused (`entrypoint.py:1093-1097`), so each request arrives in one frame.

**What exists server-side for each attachment or request (VERIFIED).**
- API arrival: `APIServer.submit_request` (`entrypoint.py:833`); `recv_time = time.perf_counter()` at `entrypoint.py:942`. Per request, not per attachment. Warmup requests also pass here (`entrypoint.py:2099-2107`), so filter them by id or time.
- Data worker: `dequeue_time` and `preprocess_finish_time` per request, only with metrics on for `dequeue_time` (`data_worker.py:1146-1175`; `request_types.py:147-162`). `upload_sizes` is per input file but has no timestamp (`request_types.py:160`).
- The default HTTP frontend is the Rust `mstar-server` (`entrypoint.py:3231-3232`, `rust_frontend.py`). The Python `recv_time` is when the request reaches `submit_request`, not when the HTTP body arrived. A grep of `rust/` found no arrival timestamp in the bridge path (UNVERIFIED in depth, see below).

**Realtime and streaming modules (VERIFIED by grep and reading).**
- `mstar/api_server/openai/serving_realtime.py` and `realtime_vad.py`: the only `time.monotonic()` uses are on the output-playback clock (`serving_realtime.py:1214, 1246, 1273`). No per-append arrival stamp.
- `mstar/streaming/` (`chunk_policy.py`, `stream_buffer.py`, `topology.py`): graph-level chunking between nodes, no wall-clock stamps.
- `mstar/validation/client_streams.py`: validation only.
- `/v1/realtime` (`openai/router.py:215-254`) serves Qwen3-ASR transcription and full-duplex conversation models, not Qwen3-Omni.

**Clock (VERIFIED).** The API server, conductor and workers all use `time.perf_counter()` on one host, and the docs state `perf_counter` is `CLOCK_MONOTONIC` and comparable across processes (`docs/mintlify/serving.mdx`, Timeline paragraph). This host reports `clock_gettime(CLOCK_MONOTONIC)` for both `perf_counter` and `monotonic` (run with the M* env Python). Realtime uses `time.monotonic`, the same clock on Linux.

**Nearest existing mechanism for chunk timestamps.** The client. `benchmark/request.py` stamps `time.monotonic()` on send and on every output chunk (`request.py:129-199`, 1682). For inputs, the study must either send each chunk as its own request (then use the API `submit_request` stamp and the profile's `first_chunk_time`) or stamp the send on the client.

### Q4. Existing benchmarks and tests that measure timing

| File | What it measures | Reusable for the chunk study? |
|---|---|---|
| `benchmark/request_timeline/report.py` | Parses `--log-stats-file` profiles into a per-request checkpoint table (mean offsets from recv, stage gaps, worker span, per-node totals) | Yes, directly. Its docstring says it reads any `--log-stats-file`. Checkpoints are the API and conductor ones; it does not give encoder start or end. |
| `benchmark/worker_phases/{server.py,phases.py,README.md,test_phases.py}` | `MSTAR_PHASE_TIMING` windows: p50/p95/mean per phase and per `step.<node>.<walk>` key; `.gpu_exec` from CUDA events (`phase_timer.py:99-113`) | Partly. Per node and walk, aggregated over windows; no request ids. Useful for GPU time per encoder step. |
| `benchmark/request.py`, `benchmark/runner.py` | Client-side TTFT per modality, E2E, output-chunk arrivals (`time.monotonic`), closed-loop runner | Yes for client-side first token and output arrival. Not for input arrival. |
| `benchmark/cosmos3/bench_stream_video.py` | TTFF and chunk gaps for streamed video output (Cosmos3) | As a pattern for output chunk timing only. |
| `benchmark/asr_realtime_bench.py` | Latency from audio send to partial result over `/v1/realtime` (`time.perf_counter` at each send, line 95-111) | Pattern for per-chunk send stamps (Qwen3-ASR, not Qwen3-Omni). |
| `benchmark/nemotron_duplex/realtime.py` | Realtime duplex client; `perf_counter` at sends and events (112, 144-153) | Pattern only. |
| `benchmark/tts_speech_bench.py` | TTFA and RTF for `/v1/audio/speech` | No (TTS). |
| `benchmark/api_output_path/{bench.py,http_e2e.py}` | CPU cost of the output path with a fake engine | No. |
| `benchmark/eval/e2e/request_path.py` | Request-path correctness suite; no timing | No. |
| `test/qwen3-omni/*.py`, `launch_nsys.sh` | Functional clients for image, audio, video, TTS. `launch_nsys.sh` runs `nsys` with `--enable-nvtx`. NVTX ranges `engine.<node>.<walk>.bs<N>` (engine.py:1893-1963) | Functional clients: no timing. nsys: GPU timeline per node and walk, no request ids. |
| `test/modular/test_worker_metrics.py`, `test_transfer_metrics.py` | Unit tests of recorders and worker metrics | Reference for building `WorkerMetrics` in a harness. |
| `test/modular/test_qwen3_omni_prefix_cache.py` | Prefix-cache behaviour (cited in the 11-25 audit; not re-read here) | Reference. |
| `perf_testing/offline_homogenous.sh` | Batch-size sweep driving `benchmark/run_benchmark.sh` | No (throughput). |

Not read in depth: `benchmark/kernels/timing.py`, `benchmark/cosmos3/bench_throughput.py`, `benchmark/eval/suites/*`.

### Q5. Hook plan (wrappers only)

**Process model (VERIFIED).**
- P0, API server: the user's `mstar serve` process, which runs `APIServer`, the data-worker threads and the message thread (`cli/main.py:_serve` calls `serve_main` in-process, `cli/main.py:259-300`; `entrypoint.py:533`).
- P1, conductor: `mp.get_context("spawn")` (`entrypoint.py:3278-3290`).
- P2..Pn, workers: spawn children of the conductor, one per rank (`conductor.py:940-980`). Engine, GPU thread and step pipeline run here.
- Rust frontend `mstar-server`: a separate binary (`rust_frontend.py:431`). Not hookable from Python.

**PYTHONPATH sitecustomize (VERIFIED by probe).**
- Spawn children inherit the parent environment and run `site` at startup. `multiprocessing/spawn.py:228` replaces `sys.path` from the preparation data after startup, so a `sitecustomize` found through `PYTHONPATH` loads first. The probe in `/tmp/audit_mstar_meas/hookprobe/` (M* env Python 3.12.3, `PYTHONPATH` set, no mstar import) printed one `sitecustomize` line for the parent and one per spawn child; the child body ran after it. Result: `/tmp/audit_mstar_meas/hookprobe/marks.txt`.
- Launcher env (VERIFIED): `_serve` changes only `CUDA_VISIBLE_DEVICES` and `MSTAR_COMPILE_LEVEL` (`cli/main.py:261-264`). `apply_process_env_defaults` uses `setdefault` (`platforms/__init__.py:79-92`). The conductor's worker target does `os.environ.update(env or {})` (`conductor.py` worker target; `env` is the MPS client env, only with `mps: true`, `mps.py:45-50`). The Rust frontend gets a copy of `os.environ` minus a few keys (`rust_frontend.py:~398-431`). `PYTHONPATH` survives all of these.
- Caveat: `/usr/lib/python3.12/sitecustomize.py` (symlink to `/etc/python3.12/sitecustomize.py`) installs the apport hook on this host. A `PYTHONPATH` sitecustomize shadows it (INFERRED from CPython `sys.path` order). The wrapper should import it or ignore it.
- `scripts/audit_vo/sitecustomize.py` (in the repo root `scripts/`, not M*) is an existing import-hook pattern (`importlib.abc`) for wrapping framework classes after import. Use the same pattern here so that importing `mstar` or `torch` is not forced at startup.

**Clock (VERIFIED on this host).** Use `time.perf_counter()`. On Linux it is `CLOCK_MONOTONIC` (verified with `time.get_clock_info`), system-wide, and the probe showed parent and child stamps on one timeline. It is the clock M* uses for `RequestTiming` and `GraphTiming`, so wrapper stamps join directly with `--log-stats` numbers. Avoid `time.time()` (CLOCK_REALTIME, can step under NTP); the vLLM-Omni shim uses it. A client on the same host can use the same clock; a remote client cannot be aligned.

**Hook table.** Each row is a wrapper on the named class method (`file:line` of the definition). Record one JSON line per event to a per-process file, buffered in memory and flushed on request finish or at exit.

| # | Process | Target (class.method) | What it gives | Clock | Overhead |
|---|---|---|---|---|---|
| H1 | P0 | `mstar.api_server.entrypoint.APIServer.submit_request` (`entrypoint.py:833`) | Per request: arrival at the Python API layer (before it, `request_id` kwarg = client id; `file_paths` for file sizes via `os.stat`); returns the client id. Filter warmups. | `perf_counter` at entry | µs per request; one `stat` per file; no device access |
| H2 | P0 | `APIServer._finalize_profile` (`entrypoint.py:1917`) (needs `--log-stats`) | Per request, after finish: `req.profile.timing` (recv, preprocess-done, ingest, first_chunk, last_chunk, finish); `req.profile.graph_timings` (per node and walk: `first_start`, `last_end`, `total_time`, `exec_count`, `forward_time`); `req.first_chunk_times` (per modality, `entrypoint.py:363, 1427`). This covers (ii), (iii) per walk and (iv). | `perf_counter` (already stored) | once per request; no device access |
| H3 | P2..Pn | `mstar.engine.engine.Engine.exec` (`engine.py:1893`) | Per batch, at GPU-thread launch: `node_name` (436), `step_context.graph_walk` (443), wire ids from `per_request_info[r].request_id` for `r >= 0` (442, `request_info.py:21-29`). Record start and end. Stash wire ids keyed by `id(batch)` for H4 and H5. Under TP2 every rank runs this; record leader ranks or dedupe. | `perf_counter` | µs per step; no tensor reads, no sync. Filter nodes (skip `thinker_decode` if volume is a problem). |
| H4 | P2..Pn | `mstar.worker.step_pipeline.StepPipeline._finish_step` (`step_pipeline.py:658`) | Per batch, after postprocess: the end time that includes the completion-event wait (`step_executor.py:858-861`). Chain and side lanes both pass here (411, 597). Join with H3 on `id(batch)`. | `perf_counter` | µs per step; adds no sync (the wait is M*'s own) |
| H5 (optional) | P2..Pn | `mstar.engine.resources.runner.StepRunner.resolve_cached_prefix` (`runner.py:147`) | Per request, per node and walk: returned `CachedPrefix.tokens` (`mstar/engine/resources/base.py:125-133`). `rid` is a worker handle; join with the wire id stashed by H3 for the batch in progress. Called on the first-chunk probe (`engine.py:1675-1698`) and the unchunked path (`_skip_cached_prefix`, `engine.py:1847`). | `perf_counter` | µs per call; no sync |
| H6 (optional) | P2..Pn | `mstar.worker.step_executor.StepExecutor.submit` (`step_executor.py:658`) | GPU-thread hand-off per step (the time work leaves the main loop). Gives queue wait before the encoder runs. | `perf_counter` | µs per step |
| H7 | client | study client | Per chunk, when sent as its own request (no input stream exists for Qwen3-Omni): send and first-byte stamps. | `perf_counter` (same host) | none |

Not recommended: `WorkerMetrics.step_done` (`worker_metrics.py:135`) (node and walk labels only, no ids) and `WorkerProfileInfo.register_end` (`profile/worker.py:63`) (keyed by handle, no wire id, batch window). `register_end` is what `--log-stats` already reports.

**Expected overhead.** Pure Python: a few perf_counter calls and dict operations per step; none touches a tensor or a stream, so none adds a device sync. The per-step cost is microseconds (UNVERIFIED magnitude). The risk is I/O: writing one line per decode step at decode rate. Buffer in memory and flush per request.

**Which nodes give which timestamp (for the four events).**
- (i) arrival of each attachment: H1 (per request) and H7 (per chunk, if separate requests). No per-attachment server stamp exists.
- (ii) vision or audio encoder completes: H4 with `node_name` in {`vision_encoder`, `audio_encoder`}. Start from H3 (GPU-thread launch, not the CPU prepare before it, which H6 can cover).
- (iii) Thinker prefill for it completes: H4 with `node_name == "Thinker"` and walk `prefill_vision` or `prefill_audio`. The last step of the walk is the prefill end.
- (iv) query arrival and first token: H1 (or H2 `recv_time`) for arrival and H2 `first_chunk_times["text"]` for the first text chunk. The "text" chunk is the first token only if text chunks are per token (INFERRED; see below).

**Other facts for the study.**
- The M* log-stats `fwd` includes GPU completion (see Q2), so do not read it as CPU launch time.
- Batched steps share one window per row (worker.py:63-100). Per-request encoder time under batching is an upper bound unless H3 records the batch size.
- Prefix hits are not logged per request. Use H5 or the per-node counters.

---

## Recommended hook table (summary)

| Hook | Process | Gives | Clock | Overhead |
|---|---|---|---|---|
| `APIServer.submit_request` | API | request arrival, file sizes | perf_counter | µs/request |
| `APIServer._finalize_profile` (log-stats) | API | per-request first/last per node and walk, first chunk | perf_counter | per request |
| `Engine.exec` | worker | per-step launch window and wire ids | perf_counter | µs/step, no sync |
| `StepPipeline._finish_step` | worker | per-step end after completion event | perf_counter | µs/step, no added sync |
| `StepRunner.resolve_cached_prefix` (optional) | worker | per-request prefix hit tokens | perf_counter | µs/call |
| `StepExecutor.submit` (optional) | worker | GPU hand-off time | perf_counter | µs/step |
| client send and receive stamps | client | per-chunk send and first byte | perf_counter (same host) | none |

Launch: set `PYTHONPATH=<dir with sitecustomize.py>` in the environment of the `mstar serve` command. `--log-stats` must be on for H2 and for the profile output. `--enable-metrics` is not needed for H1 to H7.

---

## Could not verify

1. Runtime behaviour: no server was run, so the wire-id form in `per_request_info` in worker processes (INFERRED from the `request_info.py` comment and the data worker's send), whether padded or dummy rows appear at `Engine.exec` entry, and whether rebuild rows have entries are UNVERIFIED.
2. Overhead magnitudes for H1 to H7 are UNVERIFIED (no measurement).
3. Whether `_finish_step` is reached for every encoder step on every path (speculation, side lanes, TP follower ranks) was checked only at the two call sites (step_pipeline.py:411, 597).
4. The condition under which a step has no `completion_event` (so `wait_for_step` falls back to a device-wide `torch.accelerator.synchronize`, step_executor.py:697-700) was not traced. It changes whether log-stats `all` and `fwd` include a device-wide wait.
5. Whether `first_chunk_times["text"]` equals the first generated token (INFERRED from per-chunk text emission, not traced through the result path).
6. Whether the Rust frontend forwards a request only after the full HTTP body arrives, and whether it stamps arrival (rust/ grep found no timestamp in the bridge; the bridge code was not read in full).
7. Whether the vision encoder runs once per video or in several steps. The 11-25 audit states one call per video (VERIFIED there); not re-checked here.
8. The `--log-stats` overhead (the `enable_prof` branches add `perf_counter` stamps and rx/tx info; no explicit device sync was found in the branches read, but `tensor_manager.get_rx_info`/`get_tx_info` were not read).
9. The `sitecustomize` probe did not import `mstar`, `torch` or the engine. The import-hook design (as in `scripts/audit_vo/sitecustomize.py`) was not executed against M*.
10. Whether warmup requests carry a distinguishable id (not traced; filter by time before the study starts).
11. Shadowing by `/etc/python3.12/sitecustomize.py` (apport hook) under a `PYTHONPATH` sitecustomize: INFERRED from `sys.path` order; not executed.
12. Minor: `StepRunner.resolve_cached_prefix`, `KVManager._lease` and `Engine._skip_cached_prefix` annotate `rid: str`, but the values are integer handles (`engine.py:1827`, `_resolve_hits`). Use the handle, not the annotation.
13. Docs claim: `serving.mdx:451` says `fwd` is not synced; the code (`step_executor.py:861-864`) stamps it after the completion-event wait. Treat the code as current, and the docs as stale, unless a run shows otherwise.
