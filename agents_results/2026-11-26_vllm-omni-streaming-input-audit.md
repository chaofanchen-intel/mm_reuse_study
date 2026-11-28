# vLLM-Omni streaming / incremental input audit (Qwen3-Omni focus)

> **Correction (2026-11-26, deep dive):** several statements below were superseded by [2026-11-26_vllm-omni-streaming-input-deep-dive.md](2026-11-26_vllm-omni-streaming-input-deep-dive.md). Key changes: warmups are serialized, not started per frame; prefix invariance is verified; 1280x720 is 880 tokens; the encoder cache is 62720 embeddings for all four YAMLs (CPU-derived), which also changes the batch-path capacity statement in section 3; `--enable-mm-processor-stats` is not a `vllm serve` flag; the encoder hook is `ModelState.execute_mm_encoder`; the audio gate has no stated technical reason. Read the deep dive for the corrected answers.

## Question asked

Source audit of vLLM-Omni v0.34.0 (`external/vllm-omni`, HEAD `06a28821a`) and installed vLLM 0.34.0 (`envs/vllm-omni/.../vllm`). Read-only: no source changes, no servers, no GPU.

Motivation: we study reuse of invariant multimodal parts and computing input-independent parts on arrival (camera frames at 2 fps, microphone audio at another rate, question arriving later), comparing vLLM-Omni with M*. Target model: Qwen3-Omni-30B-A3B, text output first, audio output second.

1. What streaming / incremental input exists (duplex, realtime, video stream, benchmarks, vLLM-core resumable input): models, client protocol, what is computed on arrival vs at commit/query, whether audio and frames can arrive at different rates, exact vs approximate TMRoPE positions.
2. For Qwen3-Omni: which paths work with text output and with audio output; what blocks audio output; smallest change for arrival-time encoding + incremental prefill.
3. Content-keyed reuse of an invariant input (fixed reference image/video, long system prompt) across sessions: encoder-cache lifetime and eviction, prefix cache, session-scoped state.
4. Measurement: per-chunk encoder completion and prefill progress vs query arrival; reusable benchmark scripts and what they measure.

Read first, not redone: `agents_results/2026-11-25_qwen3-omni-g0-code-audit.md` (batch video path: frame sampling, ECM, R1′, prefix defaults, prompt layout).

Marks: **VERIFIED** = I read the code that does it. **INFERRED** = follows from code I read; not executed. **UNVERIFIED** = not traced.

---

## Key finding (read this first)

1. **There are two different video input paths, and the prior audit covers only one.** The batch chat path (`/v1/chat/completions`, one `video` item per clip) is described in the 11-25 audit. The streaming path `WS /v1/video/chat/stream` sends each frame as a separate `image_pil` item with its own uuid. So on the streaming path, **the encoder cache granularity is per frame, not per clip**, and there are no `fps`/`num_frames`/temporal-patch video semantics. VERIFIED: [video_stream_base.py:231-251](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L231-L251), [serving_video_stream.py:56-112](../external/vllm-omni/vllm_omni/entrypoints/openai/serving_video_stream.py#L56-L112).
2. **The streaming video path already does arrival-time encoding and incremental prefill for frames, but only in text-only sessions.** Audio is never encoded on arrival; it is buffered and attached at query time.
3. **The `"audio" not in config.modalities` gate at [video_stream_base.py:297-299](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L297-L299) has no technical blocker that I could find.** The docs say audio-output sessions "stay on the legacy path" ([docs/serving/video_stream_api.md:117-118](../external/vllm-omni/docs/serving/video_stream_api.md#L117-L118)) but also that stage-0 prefix caching "is safe for audio-output requests" ([video_stream_api.md:147-150](../external/vllm-omni/docs/serving/video_stream_api.md#L147-L150)). The gate is a policy with no written reason. One likely cost: the warmup uses the session's `modalities` ([video_stream_base.py:1358-1359, 1412](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1358-L1359)), so audio-output warmups would also drive talker/code2wav for a 1-token request. INFERRED, not run.
4. **vLLM 0.34 core has real resumable/streaming input** (`StreamingInput`, appending prompt tokens and `mm_features` to a running request and keeping its KV). **No Qwen3-Omni path uses it**: the Qwen duplex plugin declares it unsupported, and the MRv2 profile says so explicitly.

---

## 1. Streaming / incremental input inventory

### 1a. `WS /v1/video/chat/stream` (the Qwen3-Omni incremental-prefill path)

- Endpoint: [api_server.py:2004-2018](../external/vllm-omni/vllm_omni/entrypoints/openai/api_server.py#L2004-L2018) → `QwenOmniStreamingVideoHandler` ([serving_video_stream.py:50](../external/vllm-omni/vllm_omni/entrypoints/openai/serving_video_stream.py#L50)). Only Qwen3-Omni is served (`create_streaming_video_handler` returns only the Qwen handler, [serving_video_stream.py:115-127](../external/vllm-omni/vllm_omni/entrypoints/openai/serving_video_stream.py#L115-L127)). VERIFIED.
- Protocol (client to server): `session.config` (first, once), `video.frame` {data b64 JPEG/PNG, frame_id, pts_ms}, `audio.chunk` {data b64 PCM16 16 kHz mono}, `video.query` {text, optional audio_data}, `video.done` ([video_stream_base.py:10-16](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L10-L16)). Server to client: `video.frame.ack`, `video.frames.consumed`, `response.start`, `response.text.delta/done`, `response.output_audio.delta/done`, `session.done`, `error` ([video_stream_base.py:18-27](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L18-L27)). VERIFIED.
- **On `video.frame` arrival** ([video_stream_base.py:633-752](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L633-L752)):
  - optional EVS pixel-similarity filter drops near-duplicate frames ([:645-658](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L645-L658)); `video.frame.ack` reports `accepted`/`filtered` ([:648-655](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L648-L655)).
  - PIL decode in a worker thread, uuid = md5(raw bytes) ([:702-708](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L702-L708)). VERIFIED.
  - **No encoder runs here.** The encoder runs only when the engine schedules the frame's mm item. VERIFIED (no encoder call in this file).
  - Then `_maybe_start_warmup()` ([:752](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L752)).
- **Warmup (incremental prefill), arrival-time, only if stage-0 prefix caching is on and text-only** ([:466-519](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L466-L519); gate [:297-299](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L297-L299); init check [:85-93, :322](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L85-L93)):
  - Prompt = system + last two text-only history messages + all decoded frames as `image_pil` parts ([:272-291](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L272-L291), [:253-270](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L253-L270)).
  - Runs as a `max_tokens=1` `engine_client.generate` request ([:1342-1371](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1342-L1371), [:1373-1412](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1373-L1412)). It is one new engine request per warmup, not a persistent session. VERIFIED.
  - Result: encoder runs for each new frame image item and the thinker prefills the prefix; the KV blocks stay in the **global prefix cache**, not in a per-session KV object. INFERRED (follows from the prefix cache design; the warmup comment says "completed warmup blocks stay in the prefix cache", [:551-552](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L551-L552)).
  - A new warmup is started on each new frame, and each one is cancelled when a query starts ([:521-533](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L521-L533)).
- **Audio on arrival: not encoded.** `audio.chunk` only appends bytes to `audio_buffer` ([:754-764](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L754-L764)), capped at 4 MiB ([:64](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L64)). VERIFIED.
- **Query** (`video.query`, [:766-780](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L766-L780)): takes the whole audio buffer, waits for all frames to decode, then calls `_process_query_engine` ([:907-1011](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L907-L1011)). In incremental mode all frames are used, no stride subsampling ([:1470-1487](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1470-L1487)). The prompt is history + frames + `input_audio` (WAV) + query text ([serving_video_stream.py:76-86](../external/vllm-omni/vllm_omni/entrypoints/openai/serving_video_stream.py#L76-L86)). Audio therefore sits after the frames, in the suffix, and is encoded at query time. VERIFIED.
- **Rates:** video and audio have separate buffers ([video_stream_api.md:120](../external/vllm-omni/docs/serving/video_stream_api.md#L120)). They can arrive at any rate, but audio gets no engine work until the query. Frame density is the client's job; there is no server-side resampling in incremental mode ([video_stream_api.md:72](../external/vllm-omni/docs/serving/video_stream_api.md#L72)). VERIFIED.
- **Positions:** images, not video, so there is no temporal patch merging across frames and no `second_per_grid`. Because the thinker is causal and each frame is a suffix of the previous prefix, the prefix KV of a warmup equals what a one-shot request with the same frames would compute. Positional ids for earlier tokens do not change when a later frame is appended. INFERRED; I did not trace the mrope code for image items in this audit (**UNVERIFIED**: exact TMRoPE for image items).
- **Prefix stability:** the warmup prefix changes after every turn, because history is compressed to the last two text messages ([video_stream_api.md:139-141](../external/vllm-omni/docs/serving/video_stream_api.md#L139-L141)). The warmup prefix also depends on `use_audio_in_video` (passed on every warmup and query when `use_audio_in_video` is true, [video_stream_base.py:953-956, 1397-1400](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L953-L956)), because it is a hash factor (see section 3).

### 1b. `/v1/realtime` and `/v1/duplex` (duplex sessions)

- Routes: `/v1/realtime` (and `?duplex=1`) and `/v1/duplex` ([api_server.py:2038-2086](../external/vllm-omni/vllm_omni/entrypoints/openai/api_server.py#L2038-L2086)). Commands: `input_audio_buffer.append` ([engine/duplex/commands.py:102](../external/vllm-omni/vllm_omni/engine/duplex/commands.py#L102)), `input_audio_buffer.commit` ([:131](../external/vllm-omni/vllm_omni/engine/duplex/commands.py#L131)), `response.create` ([:152](../external/vllm-omni/vllm_omni/engine/duplex/commands.py#L152)), `conversation.item.create` ([:173](../external/vllm-omni/vllm_omni/engine/duplex/commands.py#L173)). VERIFIED.
- Capability flags per model ([DuplexCapabilities](../external/vllm-omni/vllm_omni/engine/duplex/config.py)):

| Model | Turn policy | `supports_core_resumable_request` | `supports_reencode_context` | Frames | Source |
|---|---|---|---|---|---|
| Qwen3-Omni | commit only | **False** | True | `input_image` conversation items, not on audio appends | [qwen3_omni/duplex/plugin.py:511-541](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py#L511-L541) |
| MiniCPM-o 4.5 | model-native | **True** | False | attached to audio appends or sent as vision-only appends | [minicpmo_4_5/duplex/capabilities.py:29](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/capabilities.py#L29) |
| Haru, PersonaPlex, Nemotron VoiceChat | model-native | True | False | not traced | `*/duplex/capabilities.py` / `plugin.py` |
| AURA-Omni | commit only | False | False | not traced | `aura_omni/duplex/capabilities.py` |
| JoyAI-VL interaction | commit only | False | True | images | `joyai_vl_interaction/duplex/plugin.py` |

- **Qwen3-Omni duplex on arrival:**
  - `input_audio_buffer.append` only appends to a PCM buffer, and "Qwen inference starts on commit, never on an individual audio chunk" ([qwen3_omni/duplex/input.py:65-81](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/input.py#L65-L81)). VERIFIED.
  - Frames on an audio append are rejected ([input.py:67-75](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/input.py#L67-L75)). Images arrive as `conversation.item.create` `input_image` items.
  - Each commit is a fresh request with the full history re-rendered ([plugin.py:1384-1400](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py#L1384-L1400)); prefix hits come only from the shared history prefix ([qwen3_omni_duplex_mrv2.yaml:10-11](../external/vllm-omni/vllm_omni/deploy/qwen3_omni_duplex_mrv2.yaml#L10-L11)).
  - Images are summarised by a private auxiliary request, not per arrival ([plugin.py:443-468](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py#L443-L468); visual memory in [engine/duplex/visual_memory.py](../external/vllm-omni/vllm_omni/engine/duplex/visual_memory.py)).
  - **Net: no arrival-time encoding or incremental KV for Qwen duplex.** VERIFIED.
- **MiniCPM-o 4.5 duplex on arrival (the native streaming reference):**
  - Unit = 1 s of audio (16000 samples; 1600 samples per audio token, [minicpmo_4_5/duplex/policy.py:85-87](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/policy.py#L85-L87)).
  - Frames ride an audio append or go as vision-only appends ([input.py:319-328](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/input.py#L319-L328)). A vision-only append is encoded and prefilled at once ([stage0.py:438-476](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/stage0.py#L438-L476)). Each frame is 64 resampler embeds plus 2 tokens ([policy.py:88-97](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/policy.py#L88-L97)).
  - KV is a persistent paged window that survives across units ([window_kv.py:1-20](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py#L1-L20)); it requires `enable_prefix_caching=False` ([window_kv.py:156-157](../external/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py#L156-L157)). VERIFIED.
  - Audio and frames can arrive at different rates only as vision-only appends. Audio is consumed in 1 s units.
  - Positions: the official streaming design, per unit (VERIFIED from comments; positions not traced).

### 1c. vLLM 0.34 core: `StreamingInput` / resumable requests

- API: `StreamingInput{prompt, sampling_params}` ([vllm/engine/protocol.py:32-41](../envs/vllm-omni/lib/python3.12/site-packages/vllm/engine/protocol.py#L32-L41)). `AsyncLLM._add_streaming_input_request` consumes an async generator, sends each chunk as `resumable=True`, and sends an empty final request when the stream closes ([vllm/v1/engine/async_llm.py:603-700](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py#L603-L700)). VERIFIED.
- Scheduler side: `_update_request_as_session` keeps the running request's KV, drops only the last sampled token (re-prefilled), appends the chunk's prompt tokens and `mm_features` (offsets rebased), and sets the status to WAITING ([vllm/v1/core/sched/scheduler.py:1933-1975](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py#L1933-L1975)). **This is the native incremental prefill with KV retained across chunks.** VERIFIED.
- Only restriction in core: no encoder-only or diffusion models ([async_llm.py:712-716](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py#L712-L716)).
- vLLM-Omni wiring: `AsyncOmni.generate` accepts an async generator and submits it as `streaming_update` ([async_omni.py:227-232, 329-335](../external/vllm-omni/vllm_omni/entrypoints/async_omni.py#L227-L232)); [async_omni_engine.py:950-1013](../external/vllm-omni/vllm_omni/engine/async_omni_engine.py#L950-L1013); [orchestrator.py:4386-4420](../external/vllm-omni/vllm_omni/engine/orchestrator.py#L4386-L4420). VERIFIED.
- Who uses it: the native duplex plugins (MiniCPM-o, Haru, PersonaPlex, Nemotron). **Not Qwen3-Omni.** The MRv2 profile says "Resumable streaming input is unsupported by this native MRv2 profile" ([qwen3_omni_moe_mrv2.yaml:14-16](../external/vllm-omni/vllm_omni/deploy/qwen3_omni_moe_mrv2.yaml#L14-L16)). The video stream calls `generate` with a dict prompt, not a generator ([video_stream_base.py:1003-1004](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1003-L1004)). VERIFIED.

### 1d. Other realtime endpoints (not relevant to input streaming)

- `/v1/realtime/video` is **output**: streamed generated video chunks (`_WS_PATH = "/v1/realtime/video"`, [benchmarks/realtime_video/client.py:26](../external/vllm-omni/vllm_omni/benchmarks/realtime_video/client.py#L26); `save_video` at [:309](../external/vllm-omni/vllm_omni/benchmarks/realtime_video/client.py#L309)). VERIFIED.
- `/v1/realtime/robot/openpi`: robot policy messages.

---

## 2. Qwen3-Omni: what works, what blocks audio output, smallest change

### 2a. Status matrix (Qwen3-Omni)

| Path | Text output | Audio output | Arrival-time encoder | Incremental KV across inputs |
|---|---|---|---|---|
| `WS /v1/video/chat/stream`, incremental prefill | works (warmup + query) | **legacy only**: no warmup ([gate :297-299](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L297-L299)) | frames: yes (warmup). Audio: no | prefix-cache blocks (global), not per session |
| `WS /v1/video/chat/stream`, legacy | works | works | no (query-time, stride-sampled) | no |
| `/v1/realtime` Qwen duplex | works (commit only) | works (per commit) | no | no (re-encode per commit) |
| vLLM core resumable | not used / unsupported by MRv2 profile | — | would be yes | yes (core supports it, [scheduler.py:1933](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py#L1933)) |

### 2b. What blocks audio output in the streaming path

- The only code gate is `_incremental_prefill_active` ([video_stream_base.py:297-299](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L297-L299)): `supported and "audio" not in config.modalities`. VERIFIED.
- Stage-0 prefix caching is the omni prefix cache, which stores the thinker's hidden states for the talker. Thinker default `needs_full_hidden_states=True` ([core/prefix_cache/interface.py:138-145](../external/vllm-omni/vllm_omni/core/prefix_cache/interface.py#L138-L145)); the design doc lists Qwen3-Omni thinker and talker as supported ([docs/design/feature/prefix_caching.md:256](../external/vllm-omni/docs/design/feature/prefix_caching.md#L256)). VERIFIED.
- The caveat the docs give: a preempted request that resumes on a hit does not get hidden states back ([video_stream_api.md:147-150](../external/vllm-omni/docs/serving/video_stream_api.md#L147-L150)). This is a sizing issue, not a block.
- Other audio-related prefix-cache effects: `talker_first_audio` needs prefix caching **off** (a different talker-first codec path, TP1 V2 runner only, [model_executor/models/common/talker_first_audio.py:15-35](../external/vllm-omni/vllm_omni/model_executor/models/common/talker_first_audio.py#L15-L35)). With prefix caching on, the regular codec path is used. VERIFIED. This changes latency, not correctness.
- Warmup cost (INFERRED, not run): `_drive_warmup_generate` passes `output_modalities=modalities` ([:1359](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1359)) and `_prefill_context` passes `config.modalities` ([:1412](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1412)). With audio, each warmup would also run talker/code2wav for a 1-token output whose audio is discarded.
- **Conclusion: I found no hard technical blocker. The gate looks like a policy choice, and its reason is not in the code or docs. The docs' safety claim is only as good as the audio-output prefix-hit test, and I have not seen that test in `tests/`.** (`grep` for audio + warmup in `tests/entrypoints/openai_api/test_serving_video_stream.py` found only audio-in-video kwargs tests.) UNVERIFIED on GPU.

### 2c. Smallest change to enable audio output with arrival-time frame encoding

Option A (small, ~30 lines + tests). Keep frame warmup for audio sessions too, but warm with text-only modalities:

1. Relax the gate: `_incremental_prefill_active` drops the `audio` check ([video_stream_base.py:299](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L299)).
2. Warmup uses text-only output: `_drive_warmup_generate` / `_prefill_context` pass `output_modalities=["text"]` ([:1359](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1359), [:1392](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1392), [:1412](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1412)). The thinker still saves hidden states to the prefix cache (policy default), so the audio query can hit. UNVERIFIED: whether a text-only warmup saves the full-prompt hidden states the talker needs, and whether the orchestrator skips stage 1 for text-only requests (final stage selection: [omni_base.py:476-485](../external/vllm-omni/vllm_omni/entrypoints/omni_base.py#L476-L485); the async-chunk forward path was not traced).
3. Test: a text-only warmup followed by an audio query must give identical audio to a cold request with the same prompt, and `num_cached_tokens` (see section 4) must cover the frames.

Risks: the preemption caveat above; the uncertainty about hidden states from text-only warmups. Needs a GPU test (not done here).

Option B (medium, ~100–200 lines, prompt-format change). Arrival-time **audio** encoding. Audio chunks would have to enter the prefix as their own items in arrival order, interleaved with frames. Today the Qwen prompt has one `input_audio` at the end of the user turn ([serving_video_stream.py:76-86](../external/vllm-omni/vllm_omni/entrypoints/openai/serving_video_stream.py#L76-L86)). Changing that changes the prompt layout and the model's behaviour, so it needs an accuracy check. Estimate, not measured.

Option C (large). Native vLLM resumable input for Qwen3-Omni: needs a Qwen3-Omni duplex-style plugin with `supports_core_resumable_request=True` and an MRv2-compatible resumable path. The MRv2 profile says it is unsupported today ([qwen3_omni_moe_mrv2.yaml:14-16](../external/vllm-omni/vllm_omni/deploy/qwen3_omni_moe_mrv2.yaml#L14-L16)). Not sized.

---

## 3. Content-keyed reuse across sessions (fixed image, long system prompt)

### 3a. The mm key on the streaming path

- The handler sets the item uuid to md5(raw bytes) ([video_stream_base.py:702](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L702)). VERIFIED.
- vLLM uses the client uuid directly as the mm hash **when there are no hash factors**; with hash factors (`mm_processor_kwargs`, `media_io_kwargs`, model id) it hashes uuid + factors ([vllm/multimodal/processing/inputs.py:84-106](../envs/vllm-omni/lib/python3.12/site-packages/vllm/multimodal/processing/inputs.py#L84-L106)). VERIFIED.
- Consequence: the same image bytes in two sessions give the same mm key, as long as the hash factors are the same. The streaming path sends `use_audio_in_video` as a processor kwarg on every warmup and query when `use_audio_in_video` is true ([video_stream_base.py:953-956, 1397-1400](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L953-L956)). That makes the kwarg a hash factor, so **a session that toggles `use_audio_in_video` or sends audio changes its keys**. Keep it fixed across a study.

### 3b. Encoder cache (EncoderCacheManager)

- Scheduler-side, in the engine core process, keyed by mm hash ([v1/core/encoder_cache_manager.py:18-60](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/encoder_cache_manager.py#L18-L60)). It is **not session-scoped**. VERIFIED.
- Entries are referenced by request IDs. When the last reference is freed, the entry becomes "freeable" but stays in memory until allocation needs the space ([encoder_cache_manager.py:259-297](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/encoder_cache_manager.py#L259-L297)). Eviction is of the oldest unreferenced entries ([:21-60](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/encoder_cache_manager.py#L21-L60)). VERIFIED.
- Capacity is set by the scheduler's encoder cache size, in embeddings, not frames (prior audit, [encoder_cache_manager.py:369-374](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/encoder_cache_manager.py#L369-L374)). Per-frame image items are small, so many frames fit. INFERRED: a 448–1024 px frame is on the order of hundreds of embeddings; not measured.
- Reset: `AsyncOmni.reset_encoder_cache` ([async_omni.py:1318-1327](../external/vllm-omni/vllm_omni/entrypoints/async_omni.py#L1318-L1327)). I found no HTTP route for it in `vllm_omni/entrypoints`.

### 3c. Prefix cache

- Block-level, 64-token blocks for Qwen3-Omni (prior audit). A block's hash includes the mm hash of the item it overlaps (prior audit). Global, not session-scoped. VERIFIED from prior audit and [block_pool.py](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/block_pool.py).
- Eviction: freed cached blocks sit in an LRU free queue; `_maybe_evict_cached_block` runs when a block is reused ([block_pool.py:751-837](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/block_pool.py#L751-L837)). VERIFIED.
- Hit cap: at least the last token is always computed ([vllm/v1/request.py:355](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/request.py#L355)). VERIFIED.
- Session prompts do not set `cache_salt` (grep for `cache_salt` in `video_stream_base.py` returns nothing). So prefix blocks are shared across all clients whose prompts share a token prefix. VERIFIED. This matters for multi-tenant timing side channels, not for the study.
- A long system prompt is part of `_history_prefix_messages` ([video_stream_base.py:253-270](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L253-L270)). An identical system prompt gives identical leading blocks across sessions, so it hits across sessions as long as the blocks are resident. INFERRED from block hashing.

### 3d. Session-scoped state in the video stream (what is per-connection)

- Per WebSocket session, local to `handle_session`: `frame_buffer`, `frame_metadata`, `frame_pil_cache`, `frame_ready`, `audio_buffer`, `message_history`, the warmup and query tasks ([video_stream_base.py:333-340, 357-380](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L333-L380)). VERIFIED.
- Consequence: a fixed reference image sent in each session is re-decoded per session (PIL cache is per session), but the **server-side encoder and prefix caches hit across sessions** if the bytes and the prefix match. The PIL cache is dropped when a frame leaves the buffer ([:357-361, :671-672](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L357-L361)).
- Duplex: history and visual memory are per session (`QwenDuplexSessionState`, [qwen3_omni/duplex/session.py:17-38](../external/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/duplex/session.py#L17-L38)). Server caches are shared as above.

---

## 4. Measurement

### 4a. Timestamps that already exist (WebSocket video stream)

- `video.frame.ack`: sent per frame; `_send_frame_ack` at [video_stream_base.py:1489](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1489); accepted or filtered, `buffered_frames`.
- Receiver stamp: each frame's metadata carries `_receiver_received_ts_ms` and `capture_ts_ms` from the client ([video_stream_base.py:681-689](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L681-L689)). VERIFIED.
- `video.frames.consumed` (sent once, after the first engine output, [video_stream_base.py:1011 onward](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1011)): per consumed frame `receiver_received_ts_ms`, `decoded_ready_ts_ms`, and `model_selected_ts_ms`, plus `latest_pts_ms`. VERIFIED.
- Per query: `[TIMING] mode total first_text first_audio audio_chunks` log line ([video_stream_base.py:1128-1134](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L1128-L1134)). VERIFIED.
- **Missing:** no timestamps for warmup start, warmup finish, or per-frame encoder completion. Warmup failures are only logged at debug level ([:507](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L507)).
- **Cache hits:** `OmniRequestOutput.num_cached_tokens` ([outputs/__init__.py:197](../external/vllm-omni/vllm_omni/outputs/__init__.py#L197), set from prefill stats at [output_processor.py:888](../external/vllm-omni/vllm_omni/outputs/output_processor.py#L888)). It is available inside `_drive_query` and `_drive_warmup_generate` but is not sent to the client. VERIFIED.

### 4b. Per-chunk encoder timing (engine side)

- Scheduler: `EncoderCacheManager.check_and_update_cache(request, input_id)` returns whether the item was cached ([encoder_cache_manager.py:111](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/core/encoder_cache_manager.py#L111)). Wrapping it gives per-item hit/miss in the engine-core process. INFERRED that a sitecustomize on PYTHONPATH reaches that process.
- Worker: MRv2 `EncoderRunner.execute_mm_encoder(self, mm_kwargs)` covers both encoder paths ([vllm/v1/worker/gpu/mm/encoder_runner.py:150-172](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/worker/gpu/mm/encoder_runner.py#L150-L172)); `--enable-mm-processor-stats` records `encoder_forward_secs` per request ([encoder_runner.py:174-194](../envs/vllm-omni/lib/python3.12/site-packages/vllm/v1/worker/gpu/mm/encoder_runner.py#L174-L194)), but it is read through `collective_rpc`, which no HTTP route exposes (prior audit). VERIFIED for the code; the route absence is INFERRED from the prior audit.

### 4c. Prefill progress vs query arrival (proposed)

1. Client: record send time per frame and per query.
2. Server, existing: `receiver_received_ts_ms` per frame.
3. Server, add (about 10 lines): wall-clock start and end of each `_prefill_context` call with the frame count, plus `num_cached_tokens` from the output. Log a line per warmup.
4. Engine, add: the encoder hook above with request id and item uuid, so encoder time can be attributed to the warmup that triggered it.
5. Query: `video.frames.consumed` timings, `[TIMING]` log, and `num_cached_tokens` from the query output. A query "pays for its suffix" if the query's `num_cached_tokens` covers all frames.

Caveat: warmups are cancelled and restarted on every arrival ([:521-533](../external/vllm-omni/vllm_omni/entrypoints/openai/video_stream_base.py#L521-L533)), so a warmup can be aborted before it finishes. Log cancellations too, or the timings will be biased toward completed warmups.

### 4d. Existing benchmark and example scripts

| Script | What it does | Reusable here? |
|---|---|---|
| [vllm_omni/benchmarks/realtime_video/](../external/vllm-omni/vllm_omni/benchmarks/realtime_video/) (`client.py`, `serving.py`, `metrics.py`) | client for `/v1/realtime/video`: streamed **video output** (generation), raw-measurement validation | No (output path, not input) |
| [vllm_omni/benchmarks/duplex_session_metrics.py](../external/vllm-omni/vllm_omni/benchmarks/duplex_session_metrics.py) | assembles per-session metrics from `EventCollector.timing_summary`: RTF, response latency, `stream_*` ([:1-12](../external/vllm-omni/vllm_omni/benchmarks/duplex_session_metrics.py#L1-L12)) | Partly: duplex (`/v1/realtime`) only; no encoder or prefill timing |
| [vllm_omni/benchmarks/duplex_session_inputs.py](../external/vllm-omni/vllm_omni/benchmarks/duplex_session_inputs.py) | replays a clip into a duplex session with model-aware commit rules (Qwen: `server_vad` turns; MiniCPM-o: commit once) | Yes for `/v1/realtime` Qwen runs (send schedule, commit rule); not for the video WS |
| [examples/online_serving/qwen3_omni/streaming_video_client.py](../external/vllm-omni/examples/online_serving/qwen3_omni/streaming_video_client.py) | example client for `/v1/video/chat/stream`: sends frames, optional audio, a query; prints streamed text | Basis for a client; no timing output (INFERRED from its docstring and usage) |

No benchmark exists for `/v1/video/chat/stream`. Its only tests are in `tests/entrypoints/openai_api/test_serving_video_stream.py`, which is unit-level (mocked engine).

---

## What I could not verify (UNVERIFIED)

- Whether a text-only warmup saves the full-prompt hidden states the talker needs for a later audio query, and whether stage 1 is skipped for text-only requests (orchestrator forward path not traced). This decides Option A in 2c.
- The reason for the `"audio"` gate. No code comment or doc states it.
- Exact TMRoPE positions for `image_pil` items (no mrope code for image items traced).
- Encoder cache capacity for the streaming path's per-frame items at the default `max_num_batched_tokens` (not traced for this path).
- Whether `use_audio_in_video` changes anything for image items other than the hash (I only traced the hash factor).
- Whether the streaming video warmups actually hit on a GPU; no run was done (policy: no GPU in this audit).
- Whether `sitecustomize` reaches the engine-core process under vLLM-Omni's launcher (INFERRED; prior audit reached the same caveat for M*).

## Pointers for the study

- For frame-level encoder reuse with a fixed reference image: the streaming video path is the right test bed. Each frame is its own mm item and its own cache key (section 3a), unlike the batch path where one clip is one item.
- Keep `use_audio_in_video` and the frame set fixed within a study, since both are hash factors (section 3a).
- Put the `num_cached_tokens` and warmup timing logging (section 4c) in place before any run, or the result cannot be separated from the frames the warmup covered.
