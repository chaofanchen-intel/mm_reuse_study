# M* streaming input, encoder reuse, and measurement hooks (Qwen3-Omni focus)

## Question asked

Read-only source audit of M* (pinned commit `2c8ffde8f`, `external/mstar`). Context: we study whether a graph-scheduled multimodal runtime is better suited than vLLM-Omni to:

1. running the input-independent parts of a request as soon as each input arrives (camera frames at 2 fps, microphone audio at another rate, the question arriving later), and
2. reusing invariant subgraph outputs across requests (encoder outputs, KV prefixes).

Target model: Qwen3-Omni-30B-A3B (text first, audio second). Answer, with file:line citations, for:

1. what streaming or incremental input exists in M* today, per path;
2. whether Qwen3-Omni supports any of it, the smallest change to accept chunks, and whether incremental prefill is exact;
3. where a content-keyed cache of node outputs could go, and how walks decide to run;
4. measurement hooks and how to timestamp per-chunk encoder completion against query arrival.

Prior context, not redone: `2026-11-25_qwen3-omni-g0-code-audit.md`.

Constraints: read-only on `external/mstar`; no servers, no GPU. CPU-only Python checks were allowed; no weights loaded.

Sub-reports (each with its own citations and VERIFIED/INFERRED/UNVERIFIED marks):

| Sub-report | Covers |
|---|---|
| [2026-11-26_mstar-streaming-inventory.md](2026-11-26_mstar-streaming-inventory.md) | Q1: every streaming and session path |
| [2026-11-26_mstar-qwen3-incremental-exactness.md](2026-11-26_mstar-qwen3-incremental-exactness.md) | Q2: Qwen3-Omni positions, audio interleave, walk structure, smallest change, exactness |
| [2026-11-26_mstar-encoder-output-cache-hooks.md](2026-11-26_mstar-encoder-output-cache-hooks.md) | Q3: insertion points for an encoder-output cache |
| [2026-11-26_mstar-measurement-hooks.md](2026-11-26_mstar-measurement-hooks.md) | Q4: metrics, request ids, clocks, wrappers |

Marks: **VERIFIED** = read in code (lead agent re-checked the ones marked "(me)"). **INFERRED** = follows from code, not run. **UNVERIFIED** = needs a live engine or GPU.

---

## Answers

### Q1. Streaming and session paths in M*

| # | Path | Models | What arrives | Per-chunk work | State kept |
|---|---|---|---|---|---|
| P1 | WS `/v1/realtime?intent=transcription` | qwen3_asr | audio chunks | re-decodes all audio every `chunk_seconds`, new request each step | none (no KV) |
| P2 | WS `/v1/realtime` conversation | nemotron_duplex | 80 ms audio frames | encoder, LLM and talker per frame | KV, Mamba and encoder state for the call |
| P3 | WS `/generate/ws` live call | nemotron_duplex, qwen3_tts, waypoint | audio, text or actions | per request | KV per request |
| P4 | Rust `mstar-server` (default frontend) | same as P1–P3 | same | same via bridge; VAD in Rust | as P1–P3 |
| P5 | Dynamo realtime bridge | qwen3_omni (flag) | PCM | buffers; one chat request per commit | none across turns |
| P6 | Cosmos3 `session_id` / `resume_session` | cosmos3 | request-level | stores latents, not KV, across requests | latents; KV only within one request |
| P7 | openpi WS at `/` | pi05 | per call | stateless | none |
| P8 | Long transcription windows, speech sentence groups | — | — | independent requests | none |

- Only P2 and P3 (nemotron_duplex, and live-call qwen3_tts and waypoint) keep KV across chunks in the same call. The others are turn-based. (sub-report; INFERRED that the KV is retained at runtime, since only source was read.)
- **Qwen3-Omni has no client streaming path.** `/v1/realtime` closes with 1008 for models that do not serve it ([router.py:250-252](../external/mstar/mstar/api_server/openai/router.py#L250-L252), comment: "the served model has no such endpoint") (me: VERIFIED). Only Dynamo (P5) accepts streamed input for it, and only if deployed.
- **No video input stream exists** in M*. The three client streams are audio, text and actions.
- **Timestamps** are server-side per request only (`APIServer.submit_request`, `recv_time`).

### Q2. Qwen3-Omni: incremental prefill, positions, exactness

**Positions (TMRoPE):**
- Within one attachment, appending later temporal groups leaves earlier positions unchanged in the formula: temporal index = origin + t × spg × 13, and h/w do not depend on total length. CPU check: a 3551-position prefix is identical (sub-report).
- **But the real pipeline breaks this.** `spg = 2 / sample_fps`, and the sample fps is chosen from the video's total duration. A 10.0 s clip gives spg 1.0 and a 9.9 s clip gives 1.1. The per-frame resize also shrinks above about 150 s. So the text after a video depends on the video's length. (sub-report; spg code at [qwen3_omni_model.py:1110-1122](../external/mstar/mstar/model/qwen3_omni/qwen3_omni_model.py#L1110-L1122), me: VERIFIED that spg is computed there from `video_sample_fps`.)
- **Current chunking is not exact.** Splitting by attachment resets the origin of the second chunk, moving the question from position 124 to 114 and adding two sentinels (sub-report, CPU).

**Audio:**
- M* has no `use_audio_in_video`. The load path has `# TODO: support audio in video` ([qwen3_omni_model.py:919](../external/mstar/mstar/model/qwen3_omni/qwen3_omni_model.py#L919), sub-report).
- The audio encoder uses bidirectional 8 s windows (104 tokens), so 2 s chunks would not be exact. Audio positions alone are exact.

**Walk structure:**
- Each walk is `Sequential(encoder, Thinker)`, built from the whole request at intake (`data_worker.py:1012, 1030-1039`, sub-report).
- A walk is runnable when its inputs are complete ([graph/base.py:176-192](../external/mstar/mstar/graph/base.py#L176-L192), me: VERIFIED). There is also a `is_ready_for_streaming` path, active only when a node declares streaming inputs. Qwen3-Omni declares none (sub-report). INFERRED: the code already supports partial readiness, but intake still builds the walk from the whole request.
- No mechanism extends a prefill walk over time.

**Smallest change and exactness:**
- True streaming (sessions, chunked encoders, incremental Thinker prefill, growing layout in the engine): roughly 500–900 lines (sub-report estimate). **Not exact** against one-shot output, because the spg depends on total length. Exactness would need a length-independent sampling scheme, which changes outputs.
- **Smallest exact option** (untested): prefill `[system, text, video]` at upload, keyed by the whole-file digest, by extending `keyed_spans` past the video. Today the key stops at leading text, with the comment "keying past it has never been run or tested" ([qwen3_omni_model.py:1199-1201](../external/mstar/mstar/model/qwen3_omni/qwen3_omni_model.py#L1199-L1201), me: VERIFIED). At query time only the question and decode remain. About 60–150 lines in `qwen3_omni_model.py`, plus engine work that is unverified (prefix-only requests, skipping the encoder on a hit).

**Verdict:** streamed incremental prefill for Qwen3-Omni in M* is **not exact**. The whole-file prefix-cache route is exact in tokens and positions, if the keyed span is extended and tested.

### Q3. Encoder-output cache: insertion points

- **Recommended insertion (engine layer, opt-in per submodule):**
  - Hit check in `Engine._prepare_inputs` (`engine.py` ~1555–1580, beside `_skip_cached_prefix` at ~1571).
  - Replay in `Engine.exec_and_postprocess` (`engine.py` ~2606–2633): put cached tensors into `outputs.per_rid_outputs` for served request ids, so the existing store and routing run unchanged. Write after per-request postprocess, cloning outputs.
  - Key: node + walk + media content digest + model/adapter fingerprint, passed via `model_kwargs` into `walk_plan` step metadata (`qwen3_omni_model.py` ~815–860). The digest code exists (`data_worker.py:1605-1630`) but is computed only for keyed spans today.
- **Hits must replay, not skip.** A `None` from `prepare_inputs` drops the row and the node never completes. The existing cached-completion skip (`_skip_cached_prefix`) yields empty outputs, and the Thinker's `prefill_vision` reads `inputs["vision_embeds"][0]` unconditionally ([qwen3 submodules.py:413](../external/mstar/mstar/model/qwen3_omni/submodules.py#L413), me: VERIFIED). A skip would crash or corrupt.
- **Deepstack:** the entry must carry all four tensors: `vision_embeds` plus three deepstack tensors (indexes [8, 16, 24], out_hidden 2048). About 189 MB per 11520-row video, assuming bf16.
- **Vision node on a prefix hit:** the vision node always runs before the Thinker probe. A skip is output-preserving only if the probe covers the whole media span *and* the Thinker still receives grid, `seconds_per_grid` and the `vision_embeds` shape (replay, or a Thinker change). For video under current keying (`keyed_spans` = leading text, and `manager.py:1036` drops the last page), this is never true. Images can reach it.
- **Generic reach:** `PackedBlockLoopEncoder` (`components/block_loop_encoder.py`) is shared by Qwen3-Omni, Qwen3.5, Cosmos3, Muse and Phi4MM vision. `AuTEncoderSubmodule` is shared by Qwen3-Omni and Qwen3-ASR audio. BAGEL, MiniCPM-o, Whisper and Command A+ have their own classes. The engine layer covers all opt-in nodes; the component layer covers only shared ones.
- **Size:** about 400–500 lines across 8–9 files, plus tests.
- **Risks:**
  - Device memory on 48 GB cards. A host store with a byte cap is safer.
  - The key must include checkpoint identity: `preprocess_fingerprint` is the class name only ([base.py:874](../external/mstar/mstar/graph/base.py#L874)). Qwen3.5 uses a vision LoRA adapter, which must be in the key too.
  - Batch invariance of audio: AuT batches up to 8 requests.
  - Thread safety between the GPU thread and eviction.

### Q4. Measurement hooks

- **Metrics are all aggregate.** `mstar_step_seconds{component, graph_walk}` is per node and walk; `mstar_batch_size`; `time_to_first_chunk{modality}`; request preprocess, engine and delivery timings; `upload_size`; prefix counters per node. No metric has a request id label or exemplar.
- **Clock for step timing:** `perf_counter`, from hand-off to after postprocess, which waits on the step's completion event (sub-report).
- **Request ids inside the worker:** yes. `ExecutingBatch.per_request_info[handle].request_id` gives the wire id `<client_id>#<gen>` at `Engine.exec`. It is not in any metric.
- **Per-request node timing** exists only in the `--log-stats` profile: per (node, walk) total, count, first and last `perf_counter`. Batched steps give every row the batch window. So per-request encoder time is not attributable inside a batch.
- **Arrival timing:** Qwen3-Omni has no per-chunk input stream. The only server stamp is per request (`submit_request` `recv_time`). Per-attachment arrival is not recorded. The nearest mechanism is the client: send each chunk as its own request and stamp it there. The default Rust frontend does not stamp arrival.
- **Docs vs code:** the docs say log-stats `fwd` is not synced, but the code stamps `fwd_end` after the completion-event wait ([mstar/worker/step_executor.py:858-864](../external/mstar/mstar/worker/step_executor.py#L858-L864), me: VERIFIED). Treat `fwd` as including GPU completion.
- **Clock across processes:** `time.perf_counter()` is `CLOCK_MONOTONIC` on Linux and is shared by all processes on one host. The sub-report verified this with a spawn probe. It is the clock M*'s profile already uses, so wrapper stamps join directly with `--log-stats` output.
- **sitecustomize:** a `PYTHONPATH` sitecustomize loads in spawn children (probe-verified), and the launcher passes `PYTHONPATH` through. It shadows the distro apport sitecustomize.
- **Recommended wrappers (no source edits):**
  - `APIServer.submit_request`: arrival per request.
  - `APIServer._finalize_profile` (under `--log-stats`): per-request node first and last times, and first text chunk.
  - `Engine.exec`: per-step launch, with wire ids.
  - `StepPipeline._finish_step`: per-step end, after GPU completion.
  - Optional: `StepRunner.resolve_cached_prefix`: per-request prefix hits.
  - Overhead: microseconds per step with no device sync, if writes are buffered.

---

## Cross-check with vLLM-Omni (for the comparison)

- vLLM-Omni's streaming video path sends **each frame as an image item**. Its position rule for images is prefix-stable: appending an image does not change earlier positions (`2026-11-26_tmrope-image-items-prefix-invariance.md`, CPU-verified). So per-frame incremental prefill is exact there.
- M*'s video path cannot do this, because the temporal step depends on the video's total duration (Q2).
- This is a real difference between the two runtimes for the frame-stream use case, and it favours vLLM-Omni's per-frame design on exactness. It does not say which runtime is faster; that needs a GPU run.

## Implications for the study

1. Do not pursue true streaming in M* for Qwen3-Omni if the goal is exact outputs. Two options remain:
   - **Upload-time prefix** (60–150 lines, untested): keyed by whole-file digest, extending `keyed_spans` past the video. Gives exact positions, and query-time work only for the question and decode.
   - **Encoder-output cache** (400–500 lines): helps only images at present, since video spans are never covered by the prefix.
2. The measurement design should use client-side arrival stamps (one request per chunk), wrapper stamps on the shared `perf_counter` clock, and `--log-stats` for per-node totals. Per-request encoder time inside batches needs a wrapper on `Engine.exec`.
3. Keep `graph/base.py` `is_ready_for_streaming` in mind: it is the existing hook if streaming inputs are ever declared for Qwen3-Omni.

## Could not verify

- KV retention across steps for P2 and P3 at runtime (source only).
- Qwen3-ASR prefix-cache hits across P1 steps.
- Rust versus Python parity (P4).
- Dynamo deployment (P5).
- The upload-time prefix route: keying past the video is untested, and engine prefix-only requests are unverified.
- Live engine: `fwd` timing, the spawn-child sitecustomize under a real launch, and the `vision_embeds` replay.
- Some `test/modular` and `test/rust` realtime tests were not read.
- The 189 MB deepstack figure assumes bf16 and was not measured.
