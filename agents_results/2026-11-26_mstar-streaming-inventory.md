# M* streaming, realtime and session input paths: source inventory

- Date: 2026-11-26
- Repo: `/home/chaofan/workspaces/mm_reuse_study/external/mstar`, commit `2c8ffde8f` (confirmed with `git rev-parse HEAD`).
- Paths in this report are relative to the repo root unless shown as absolute. `L` = line number at that commit.
- Method: read-only source reading and grep. No server started, no GPU used, no file under `external/` changed. Scratch dir `/tmp/audit_mstar_inv/` created, not needed.
- Marks: **VERIFIED** = read in code. **INFERRED** = follows from code, not run. **UNVERIFIED** = not checked in this pass.

## Question asked

Inventory every streaming, realtime and session input path in M*. For each: (a) models served; (b) client protocol (endpoint, messages, field names, rates, turn delimiting); (c) work per arriving chunk versus work at commit or query time; (d) state kept across chunks; (e) whether audio and video can arrive at different rates and how they are aligned. State whether Qwen3-Omni is served by any streaming path, and which paths are true incremental prefill with KV kept across chunks versus turn-based re-prefill at commit.

## Scope

Commit `2c8ffde8f`. Read, in full or in the relevant ranges:

- `mstar/api_server/openai/serving_realtime.py` (transcription and conversation intents, `_run_step`)
- `mstar/api_server/openai/realtime_vad.py` (Silero, Smart Turn, `TurnSegmenter`)
- `mstar/api_server/rust_frontend.py` (bridge, `realtime_facts`, `_live`, `_end_of_turn`) and the Rust port headers `rust/server/src/realtime/{mod,vad,conversation}.rs`, `rust/server/src/ws/mod.rs`
- `mstar/streaming/` (all four modules: `__init__.py`, `stream_buffer.py`, `chunk_policy.py`, `topology.py`)
- `mstar/model/cosmos3/sessions.py`, `mstar/model/cosmos3/tests/test_sessions.py`, `mstar/model/cosmos3/tests/test_serving.py`, plus the session parts of `cosmos3_model.py`, `submodules.py`, `config.py` and the tests `test_windowed.py` and `test_windowed_decode.py`
- `mstar/model/minicpm_o/minicpm_o_model.py`, `mstar/model/nemotron_duplex/nemotron_duplex_model.py`, `mstar/model/qwen3_omni/qwen3_omni_model.py`, `mstar/model/qwen3_tts/qwen3_tts_model.py`, `mstar/model/waypoint/waypoint_model.py`
- `mstar/graph/base.py`, `mstar/model/base.py` (client stream hooks, `turn_tokens`, `walk_plan`)
- `mstar/validation/client_streams.py`, `mstar/communication/communicator.py`, `mstar/communication/rust_communicator.py`, `mstar/client/client.py`
- `mstar/api_server/openai/router.py`, `mstar/api_server/entrypoint.py` (`/generate/ws`, `submit_request`, `stream_input`), `mstar/api_server/data_worker.py` (client-stream intake), `mstar/worker/stream_buffers.py`
- `mstar/integrations/dynamo/bridges.py` and `worker.py` (realtime surface)
- `mstar/api_server/openai/adapters.py` (realtime flags, Qwen3-ASR and Qwen3-Omni adapters), `mstar/model/qwen3_asr/qwen3_asr_model.py` (prompt layout)
- `mstar/model/components/fastconformer_stream.py` (header), `mstar/engine/windowing.py` (header), `mstar/engine/chunking.py` (header), `mstar/worker/chunk_loops.py` (header), `mstar/api_server/openpi.py` (header), `mstar/api_server/openai/serving_transcriptions.py` (window requests), `mstar/api_server/openai/speech_chunking.py` (header), `mstar/api_server/streaming_response.py` (header)
- Grep for `session`, `stream`, `chunk`, `realtime`, `append`, `commit` across `mstar/`. Hits checked by name and set aside: `mstar/utils/streams.py` (CUDA stream handling, unrelated), `mstar/engine/resources/kv/transfer.py` `session_id` (transfer-engine id, unrelated), `source_session_id` in `mstar/graph/base.py:27` (transport endpoint id, unrelated).
- Context read first: `agents_results/2026-11-25_qwen3-omni-g0-code-audit.md` (only partly, for background).

## Findings

### A. Client-facing input paths (summary table)

| # | Path and endpoint | Models served | Per arriving chunk | At commit or query | State across chunks | KV kept across chunks? |
|---|---|---|---|---|---|---|
| P1 | `WS /v1/realtime?intent=transcription` | `qwen3_asr`, `qwen3_asr_realtime` only | CPU resample; Silero VAD per 512-sample frame if `server_vad`; a whole-item ASR engine step every `chunk_seconds` (default 2 s) | `commit`: final step over the tail, then `committed` and `completed` events | API-process audio list (`heard`), text hypothesis; no engine state | **No**: each step is a new request that re-encodes and re-prefills all audio so far |
| P2 | `WS /v1/realtime` (conversation, GA events) | `nemotron_duplex` only | per 80 ms frame: one Encoder step, one LLM decode step; per text token one Talker and codec step | none (`input_audio_buffer.commit` is acknowledged, not acted on) | encoder conv and attention window, audio and mel tails, nano KV, Mamba state, talker KV, codec state; all per call | **Yes** |
| P3 | `WS /generate/ws` live call, or `POST /generate` with `end_input:false` (generic live-call route) | `nemotron_duplex` (audio files), `qwen3_tts` (text), `waypoint` (action rows) | per append: items to client streams, then consuming steps | `end_input` | per request | **Yes** for all three |
| P4 | Rust frontend (`mstar-server`, the default frontend) | same as P1 to P3 | same, through the bridge | same | same | same as P1 to P3 |
| P5 | Dynamo realtime bridge (`mstar/integrations/dynamo`) | `qwen3_omni` (adapter flag only) | buffers PCM in the bridge; no engine work | `commit`: one chat request for the utterance | none across turns | **No** (turn-based) |
| P6 | Cosmos3 windowed sessions: `session_id`, `resume_session`, `end_session` (a request, not a stream) | Cosmos3 family, windowed mode only | per window: DiT denoise, commit, decode | request end: last window latents stored | `SessionStore` of latents across requests (TTL, LRU) | **Within one request** in `kv` window mode (across generated windows). **Not across requests** (latents only) |
| P7 | WebSocket at `/` (openpi policy) | any model with an `openpi:` block (`configs/pi05_droid.yaml` is the only such config in the repo) | one observation, one reply | one policy call | none | No |
| P8 | Non-streamed chunking: long transcription windows, speech sentence groups | transcription and TTS models | none (each window or sentence group is its own request) | per request | none | No |

Internal streams (not client input) are in section C.

### B. Qwen3-Omni: is it served by any streaming path?

**No client-streaming path serves Qwen3-Omni.** Evidence:

1. No client-stream connection. The model's partition topology has only internal connections: `thinker_states`, `thinker_mask` (Thinker to Talker) and `codec_tokens` (Talker to Code2Wav), `mstar/model/qwen3_omni/qwen3_omni_model.py:780-805`. The name `CLIENT` and `ClientAudio` do not appear in the model. **VERIFIED**.
2. No `process_stream_chunk`, no `turn_tokens`, no `turn_idle_steps` override in the model. `process_stream_chunk` is overridden only by `qwen3_tts` and `waypoint` (grep over `mstar/`). **VERIFIED**.
3. `Qwen3OmniAdapter` sets `supports_realtime = True` (`mstar/api_server/openai/adapters.py:478`). Nothing in the standard server reads that flag. The only consumer in the repo is the Dynamo worker (`mstar/integrations/dynamo/worker.py:149`). **VERIFIED** (grep).
4. The standard `/v1/realtime` route checks `api.client_audio` and then `supports_realtime_transcription` (`mstar/api_server/openai/router.py:222-254`, `_resolve` at `router.py:89-104`). For Qwen3-Omni both fail, so the socket is accepted and closed with code 1008 and reason "does not support this endpoint". **VERIFIED** by reading the branch; not executed.
5. Only with Dynamo deployed: P5 (commit-per-request, no KV across turns). **VERIFIED** (code); deployment not checked.
6. The Qwen3-Omni chat path is one-shot per request; `/generate/ws` takes it, but no live append (`takes_stream_input` is False without client streams, `entrypoint.py:601`). **VERIFIED**.

Internal streaming inside one Qwen3-Omni request is real: Thinker prefill, one step per attachment (`qwen3_omni_model.py:207-221`, `_thinker_steps`), chunkable Thinker prefill whose chunks the Talker consumes and extends its KV with (`qwen3_omni_model.py:480-493` and `_thinker_to_talker_policy` at `163-170`). This is intra-request, driven by the client's one request, and takes no client chunks. **VERIFIED**.

### C. Internal streams and chunked prefill (not client input)

- Qwen3-Omni: Thinker to Talker (`thinker_states`, `thinker_mask`, `FixedChunkPolicy(1)`, `allow_partial`, `continue_after_done` for `talker_decode`, `qwen3_omni_model.py:163-170, 784-797`), Talker to Code2Wav via `config.code2wav.chunk_policy` (`798-803`). Talker walk choice `_talker_walk` (`173-179`). **VERIFIED**.
- Nemotron duplex: `audio_frame` (Encoder to LLM) and `new_token` (LLM to Talker), both `FixedChunkPolicy(1)` (`nemotron_duplex_model.py:491-512`). **VERIFIED**.
- Qwen3-TTS: Talker to Codec `codec_tokens` with `ScheduledLeftContextChunkPolicy` (`qwen3_tts_model.py:728-735`). **VERIFIED**.
- MiniCPM-O: `tts_code` from main partition to Token2Wav, `LeftContextChunkPolicy(chunk=HOP, left_context=LEAD_SILENCE)`, `by_value=True` (`minicpm_o_model.py:462-473`). **VERIFIED**. The model docstring says "Half-duplex chat only: the duplex/streaming-session machinery upstream is not ported" (`minicpm_o_model.py:23`). **VERIFIED**.
- Cosmos3 windowed: `window_latents` from default partition to the window decoder, `FixedChunkPolicy(1)` (`cosmos3_model.py:943-958`). **VERIFIED**.
- Waypoint: single partition; the only stream is the client's actions edge (`waypoint_model.py:273-283`). **VERIFIED**.
- Engine chunked prefill (`mstar/engine/chunking.py`, `mstar/worker/chunk_loops.py`): a step's prompt tokens are cut to a budget; the chunk Loop counts tokens (`chunk_loops.py` docstring). Used by several models (Qwen3-Omni `qwen3_omni_model.py:480-493`, Cosmos3, others). **VERIFIED** (headers); this is not a client stream.
- Parakeet and Nemotron-ASR: streaming-architecture encoder but "run here over whole clips" (`mstar/model/parakeet/parakeet_model.py` docstring). **VERIFIED**.

### D. Details per path

#### P1. `/v1/realtime?intent=transcription` (Qwen3-ASR)

(a) Models: `qwen3_asr` and `qwen3_asr_realtime` (1.7B and 0.6B, `mstar/model/registry.py:45-46, 168-171`), through `Qwen3ASRAdapter` (`adapters.py:1430`, `supports_realtime_transcription = True` at `1440`, `max_audio_seconds = 1200.0` at `1441`). Other adapters leave the flag False (`adapters.py:325`), so the route returns 1008 for them (`router.py:244-254`). **VERIFIED**.

(b) Protocol (`serving_realtime.py` docstring `1-57`, `RealtimeTranscription.handle` `673-737`):
- Client events: `transcription_session.update` with `session.input_audio_transcription{model,language,prompt}`, `input_audio_format`, `turn_detection`, `mstar{chunk_seconds, unfixed_chunks, unfixed_tokens}`; `input_audio_buffer.append {audio: base64}`; `input_audio_buffer.commit`; `input_audio_buffer.clear`. **VERIFIED**.
- Input formats: default `pcm16` at 16 kHz; `g711_ulaw` and `g711_alaw` at 8 kHz; GA `{"type":"audio/pcm","rate":N}` for N in `REALTIME_RATES` (8000 to 48000, `serving_realtime.py:114`), resampled to 16 kHz on arrival. **VERIFIED**.
- Defaults: `chunk_seconds` 2.0, minimum 0.5 (`:121, :127`); `unfixed_chunks` 2; `unfixed_tokens` 5. Item cap `min(300 s, adapter max)` = 300 s (`:130`, `RealtimeTranscription.__init__` at `:629-638`). **VERIFIED**.
- Server events: `transcription_session.created/updated`, `conversation.item.input_audio_transcription.delta`, `mstar.transcription.partial` (whole hypothesis, `audio_seconds`), `input_audio_buffer.committed`, `...transcription.completed`, `input_audio_buffer.speech_started/stopped` (with `audio_start_ms` / `audio_end_ms`) under server VAD. **VERIFIED**.
- Turn delimiting: the client's `commit`, or a server VAD speech end, which is applied as a commit (`apply`, `serving_realtime.py:738-776`). **VERIFIED**.

(c) Per arriving chunk:
- Resample on arrival (`RealtimeSession.resample`, `serving_realtime.py:218-222`; `append` `:260-262`).
- Server VAD (if on): `TurnSegmenter.feed` scores every 512-sample frame with Silero on CPU (`realtime_vad.py:36, 334-345`); off the event loop (`serving_realtime.py:727-735`). **VERIFIED**. `semantic_vad` asks Smart Turn at pauses (`realtime_vad.py:380-395`; in Rust through the bridge `rust_frontend.py:757-775`). **VERIFIED**.
- Every `chunk_seconds` of audio, `take_chunk` (`serving_realtime.py:270-286`) hands a chunk to `step` (`:858-885`). `step` appends it to `heard` and calls `_run_step` (`:537-572`): it writes a WAV of `audio_so_far()` (`:295`) under the upload dir, submits one ordinary request (`api.submit_request(..., streaming=False)`, `:548-558`), and collects the text. The prefix passed to the model is the stable part of the previous hypothesis, once `chunk_id >= unfixed_chunks` (`:863`), via `Qwen3ASRAdapter.realtime_step_request` (`adapters.py:1463-1471`, `assistant_prefix`). **VERIFIED**.
- Cost: each step decodes the whole item again; the docstring says "an item's work grows with the square of its length" (`serving_realtime.py:60-68`). **VERIFIED** (docstring); the quadratic total is INFERRED from the per-step whole-audio request.

(c') At commit: `commit()` (`serving_realtime.py:778-792`) flushes the tail (`step(final=True)`), emits `committed`, `completed`, and clears the session.

(d) State: `RealtimeSession` (`:174-303`) holds `pending` (audio not yet in a chunk), `heard` (every chunk decoded so far, all of the item), `raw_hypothesis`, `emitted`, resampler, segmenter, `odd_byte`. The engine keeps nothing between steps: the WAV is deleted in `finally` (`:563-565`). The prompt puts the system context and the audio pad before the assistant turn (`qwen3_asr_model.py:212-229`). **VERIFIED**.

(e) Timing: no client timestamps. The server keeps a sample clock: `stream_samples`, `clock_base`, `clock_in` (`serving_realtime.py:174-212`), realigned after a commit or clear (`:239-246`). Speech events use it (`realtime_vad.py:287-300`). Partial events carry `audio_seconds` from sample counts (`:867-870`). No video.

Not verified: whether M*'s prefix cache hits on the system and header tokens before the audio. The audio tokens change every step, so only that header could hit. **UNVERIFIED**.

#### P2. `/v1/realtime` conversation (full duplex, `nemotron_duplex`)

(a) Models: only models that declare a `ClientAudio` stream and `turn_tokens`. In the repo that is `nemotron_duplex` (`nemotron_duplex_model.py:491-503, 675`). The router sends the request here when no `intent` is given (`router.py:222-236`); with more than one audio stream, or more than one output channel, it refuses. **VERIFIED**.

(b) Protocol (`serving_realtime.py:967-1438`, docstring `:967-985`):
- Client events: `session.update` (`type:"realtime"`, `instructions`, `audio.input.format`, `audio.input.turn_detection` = `server_vad` or `semantic_vad`, `audio.output.format`, `audio.output.voice`); `input_audio_buffer.append {audio}`; `input_audio_buffer.commit` (acknowledged only, `:1079-1082`); `input_audio_buffer.clear` (`:1083-1090`); `conversation.item.create` with `input_audio` parts (`:1167-1184`). Unsupported events (`response.create`, `response.cancel`, and others) get an `unsupported` error (`:940-944`). **VERIFIED**.
- Input: any `REALTIME_RATES` rate, resampled to the stream's rate (16 kHz, `nemotron_duplex_model.py:499-503`, `ClientAudio(stt.sample_rate, frame_samples=frame)`); `frame` is hop times subsampling, 80 ms (`fastconformer_stream.py:1-2`). **VERIFIED**.
- Output: `session.created`, `response.created`, `response.output_audio.delta` (base64 PCM16 in the output format, default 24 kHz via `OPENAI_PCM_RATE`, `:109`, resampled from the model's rate `:1159-1165`), `response.output_audio_transcript.delta`, `response.done`. **VERIFIED**.
- Turn delimiting: the model, not the client: text `turn_tokens` (BOS and EOS, `nemotron_duplex_model.py:675-680`) open and close a reply; `turn_idle_steps = 25` (80 ms steps, `:682`) closes a reply with no end mark. `_on_text` (`serving_realtime.py:1312-1359`), `_play` (`:1375-1393`), `_end_if_idle` (`:1394-1405`). **VERIFIED**.
- Pacing: `_clock` pads silence when the client is quiet, at most `MAX_PAD_SECONDS` (30 s) per pause, so the model keeps stepping (`serving_realtime.py:133, 137, 1236-1253`). **VERIFIED**.

(c) Per arriving append:
- `append` decodes base64, checks length, opens the call on the first append (`open` at `:1255-1275`: `submit_request(streaming=True, end_input=False)`), resamples and calls `api.stream_input(key, samples={stream: audio})` (`_put` `:1208-1234`; `entrypoint.py:1029-1120`). **VERIFIED**.
- Data worker: `_process_stream_input` (`data_worker.py:1180-1212`) takes `chunk.samples` (`:1202-1203`), then `_framed(..., per_frame=True)` (`:1265-1297`) cuts whole 80 ms frames; the remainder waits in `call.tails` (`:671`). Each frame is one item. **VERIFIED**.
- Engine, per item: the Encoder node runs one step per item (`encode` loop, `nemotron_duplex_model.py:421-430`, `FixedChunkPolicy(1)` at `:501-502`); the emitted frames feed the LLM decode loop, one frame per step (`:440-462`); each agent text token feeds the Talker loop, one step per token (`:464-475`). **VERIFIED** (structure); runtime per-step counts INFERRED.

(d) State, all per call and freed at call end (`declare_state`, `nemotron_duplex_model.py:318-367`; `kernel_settings` `:369-384`):
- Encoder: `ENCODER_STATE` (conv tails per layer), `ENCODER_FRONT` (newest audio and mel tails), `ENCODER_KV` (a bounded window, `Window(left_context+1)`), `ENCODER_STREAM` (host counts of samples and mel frames).
- LLM: `NANO_KV` for the attention layers, kept for the call: "a session holds its prompt and every frame for life (no eviction)" (`:374-377`); `MAMBA_STATE` (SSM and conv state, fp32 SSM); `NANO_SAMPLER`.
- Talker: `TALKER_KV` (two labels, `main` and `uncond` for CFG), `TALKER_PREV`, `CODEC_STATE`.
- Limits: `max_session_s = 160` (`:256`), enforced as the stream's `max_items` in samples (`:499-503`, `ClientStreamLimits`); `DEFAULT_SESSIONS = 32` concurrent (`:248`). **VERIFIED**.

(e) Timing: the client sends no timestamps. Server side, the playback clock is wall time (`played_until`, `_put` `:1208-1220`; `_clock`). Output chunks carry `metadata["step"]` (the loop iteration) and `turn` marks, so text and audio pair by step (`data_worker.py:1392-1402`; `_on_audio` `serving_realtime.py:1368-1374`). Only one audio stream, so no cross-stream alignment is needed. **VERIFIED**.

Rust port: `rust/server/src/realtime/conversation.rs` (header lines 1-12) ports this class over the bridge's `live` operations. Parity not diffed. **UNVERIFIED** parity.

#### P3. Generic live call: `/generate/ws` (and `POST /generate` with `end_input:false`)

(a) Models with client streams: `nemotron_duplex` (audio files and, via the realtime route, samples), `qwen3_tts` (text), `waypoint` (action rows). **VERIFIED**.

(b) Protocol (`entrypoint.py:2445-2480` docstring):
- Opening message: `{request_id, ...fields of /generate, end_input:false}`. Without `end_input:false` a model with client streams ends its input at open (`entrypoint.py:2545-2546`). **VERIFIED**.
- Append: `{request_id, append:{text, files:[{name,data,stream}], model_kwargs}}` (`files[].stream` names the client stream a file is for); end: `{request_id, end_input:true}` or a list of stream names. Messages are JSON text or msgpack binary. **VERIFIED**.
- Replies: `{request_id, modality, data, metadata}` frames, then `{request_id, finish:true}`; a rejected append is `{request_id, error}` with status 400 and ends the call. **VERIFIED**.
- Parsing and routing: `entrypoint.py:2655-2700` calls `api.stream_input` (`:1029`). **VERIFIED**.
- The Python SDK (`mstar/client/client.py`) has no live-call or append method. Its `generate` (`:132`) and `stream` (`:198`) are one-shot. **VERIFIED** (grep).

(c) Per append, by model:
- `nemotron_duplex` (files): `load_audio` decodes to 16 kHz and records `source_sample_rate` (`nemotron_duplex_model.py:616-644`). `_client_audio_items` refuses an append whose `source_sample_rate` differs from the stream's rate (`data_worker.py:1239-1262`, rate check `:1257-1262`). So a file appended live must be 16 kHz. VERIFIED (code); runtime refusal INFERRED.
- `qwen3_tts` (text): `process_stream_chunk` tokenizes the append alone, one `text_in` item per token (`qwen3_tts_model.py:1011-1027`; comment says tokenizing per append matches the whole text when the cut falls before a space or punctuation). Live text is refused unless `non_streaming_mode: false` and no in-context clone (`:950-956`). The Talker runs one decode step per text token (`talker_decode` loop `:634-650`; `text_in` `FixedChunkPolicy(chunk_size=1, continue_after_done=True)`, `:723-727`). **VERIFIED**.
- `waypoint` (action rows): `process_stream_chunk` turns `model_kwargs.actions` into one float32 row per action (`waypoint_model.py:392-399`, `_action_rows`). One action is one rollout step: `dit.generate_frame` (four denoise passes and a committing cache pass) then the TAEHV decode (`waypoint_model.py:1-25, 229-271`). **VERIFIED**.

(d) State:
- `qwen3_tts`: `TALKER_KV` grows one position per decode step (`qwen3_tts_model.py:538, 1174`); codec state per call (`:531`). The prefill runs once at open with the text before the stream (`:1003`). **VERIFIED**.
- `waypoint`: the world is the ring KV cache, an engine resource with one ring slot per request id (`waypoint_model.py:170-215`; `ring/manager.py:346-348` claims, `:179-181` frees). The encoder runs only at prime (`:229-240`). **VERIFIED**.
- `nemotron_duplex`: as P2 (d). **VERIFIED**.

(e) Timing: no client timestamps; order is item order. Waypoint: the action index is the frame index, tracked in host state (`waypoint_model.py:252-254` comment). Output rate reported as `inference_fps` (`waypoint_model.py:522-530`; config 147). **VERIFIED**. The raw-frame count per action (temporal compression) was not confirmed. **UNVERIFIED**.

#### P4. Rust frontend (`mstar-server`)

- The default frontend is the Rust server; `--python-frontend` opts out (`mstar/api_server/entrypoint.py:3205-3240`). **VERIFIED**.
- The bridge (`rust_frontend.py:1-40` protocol docstring) carries `live` operations (`_live`, `rust_frontend.py:676-755`) for `/generate/ws` and realtime appends (`audio` float32 samples for the `ClientAudio` stream), `end_of_turn` for `semantic_vad` (`:757-775`), and `windows` for transcription uploads.
- `realtime_facts` (`:348-383`) passes Silero and Smart Turn file paths, ONNX Runtime library, and the tokenizer; `conversation_facts` (`:326-346`) passes the stream name, rate, frame size, idle steps and silence threshold.
- Rust ports: `rust/server/src/realtime/mod.rs` (transcription, header `1-14`), `vad.rs` (Silero and Smart Turn on ONNX Runtime via `ort`, header `1-20`), `conversation.rs` (header `1-12`). Server VAD and resampling run in Rust; the semantic-VAD model call goes back through the bridge. **VERIFIED** (headers); parity with Python **UNVERIFIED**.

#### P5. Dynamo realtime bridge (`qwen3_omni` among others)

(a) Served only where the adapter flag `supports_realtime` is True (`adapters.py:290, 478`) and Dynamo is installed. In the repo the flag is True only for `Qwen3OmniAdapter` (grep), and `test/modular/test_dynamo_bridges.py:148-150` asserts that. **VERIFIED**.

(b) `RealtimeBridge` (`mstar/integrations/dynamo/bridges.py:565-830`, class docstring `:565-580`): `session.update` is refused unless audio format is `audio/pcm` at 24000 (`:762`) and `turn_detection` must be null (`:774`). Client events: `input_audio_buffer.append`, `.commit`, `.clear` (`:781, :791, :817`). Output: `input_audio_buffer.committed`, `response.*` events. **VERIFIED**.

(c) Per append: base64 decode into a `bytearray` (`:781-790`). No engine work. **VERIFIED**.

(c') Per commit: `_utterance_args` builds one chat-shaped body with one user message holding the WAV as `input_audio` (`:603-640`), then `_run_turn` submits one engine request (`:645`). Turns run in commit order, with `max_concurrent_turns = 2` (`:584`). **VERIFIED**.

(d) State: none across turns. The body has one user turn, so the prompt has no history, and each turn is a new request, so no KV is kept. **VERIFIED** (code). On commit the bridge keeps only the odd trailing byte, if any (`pcm = pcm[whole:]`, `:804`); `clear` empties it (`:817-819`).

(e) No video. Audio at 24 kHz only. **VERIFIED**.

#### P6. Cosmos3 windowed sessions (`session_id`, `resume_session`, `end_session`)

(a) Models: the Cosmos3 family in windowed serving (`cosmos3_model.py:629` `_windowed_serving_enabled`, `config.py:415-425`). **VERIFIED**.

(b) Protocol: a generation request, not a stream. Accepted kwargs at `cosmos3_model.py:180-183` (`window_mode` kv or chained, `window_frames`, `overlap_frames`, `context_frames`, `stream_video`, `session_id`, `resume_session`, `end_session`, `session_timeout_s`). Parsing at `cosmos3_model.py:1776-1800`: `resume_session` requires `session_id`; `session_timeout_s` capped at `session_timeout_max_s` (3600). Resume grows the schedule by a pinned head of at least 2 latent units (`:1800-1823`). **VERIFIED**. The example client is `examples/cosmos3_stream_video_ws_client.py` (docstring `1-12`): output windows arrive as they finish; kv mode is marked experimental in the example's argument help.

(c) Per window (output side): DiT denoise, commit, and the streaming decoder (`submodules.py` `Cosmos3VAEDecoderARSubmodule`, `:3091-3230`). **VERIFIED**. Client input between windows does not exist.

(d) State:
- `SessionStore` (`mstar/model/cosmos3/sessions.py:29-138`): bounded, LRU, pins a session to its live request (`begin` `:62`), `get` promotes (`:77`), `put` stores and unpins (`:87`), `end` forgets (`:99`), `release` unpins a removed request (`:103`), idle TTL (`:131-133`), eviction of idle sessions only (`:135-138`). **VERIFIED**.
- DiT side: the rollout's last window stored as clean latents, `window_latents.detach().clone()` (`submodules.py:1019-1029`; `_sessions` created at `:482`). A resumed request re-encodes the stored tail as clean head conditioning (`_session_tail`, `submodules.py:1000-1017`), so **no KV pages are kept across requests**. **VERIFIED**.
- Decoder side: its own `SessionStore` (`submodules.py:3123`) holding the latent context tail (`ar_tail`), never the decoder cache (docstring `:3110-3115`; store at `:3223-3226`). **VERIFIED**.
- Within one request in `kv` mode: each window's clean K/V is appended to the cache, and context beyond the horizon is released at each commit (`submodules.py:923-926`; retention policy in `mstar/engine/windowing.py` header `1-10`). This is KV kept across generated windows of one request, not across client chunks. **VERIFIED** (comment and header); runtime **UNVERIFIED**.
- `chained` mode re-pins the previous tail as clean conditioning and never touches committed state (`submodules.py:923-926`). **VERIFIED**.

(e) No client audio or video input; window index and frame count only.

Tests (what they cover; none were run):
- `mstar/model/cosmos3/tests/test_sessions.py` (84 lines) covers `SessionStore` only: LRU promotion, live pins and single writer, release, TTL expiry (live sessions do not expire), `end`. **VERIFIED**.
- `mstar/model/cosmos3/tests/test_serving.py` has no session-behaviour test. Its only mention is an accepted-keys check that includes `session_timeout_s` (`test_serving.py:757-770`). **VERIFIED**.
- Request-level session behaviour is in `test_windowed.py:700-725` (parsing, resume schedule growth, rejection cases) and `test_windowed_decode.py:186-212` (resumed decode matches the re-decode path, stored context shape). **VERIFIED** (read); not run.

#### P7. openpi WebSocket (`/`)

- `mstar/api_server/openpi.py` docstring (`1-20`): the client opens a socket at the root, reads one metadata frame, then sends one msgpack observation and waits for the actions reply, one call at a time. Images arrive decoded, not as files. **VERIFIED**.
- Route `entrypoint.py:2997-3005`. Stateless per call (no session object). **VERIFIED**.
- Models: any with an `openpi:` block in its serving config. `configs/pi05_droid.yaml` is the only config in the repo containing that key (grep). **VERIFIED**.

#### P8. Non-streamed chunking (not streaming input)

- Long transcription: `WindowPlanner` (`serving_transcriptions.py:210`) and each window is a separate `submit_request` (`:425`), sequential or parallel (`:572, :657, :863`). **VERIFIED**.
- Speech sentence groups (`mstar/api_server/openai/speech_chunking.py` header `1-10`): inputs over `speech_chunk_min_chars` are split and "synthesized as separate requests". **VERIFIED** (docstring); each being its own request **INFERRED** (`serving_speech` not read).

### E. Qwen3-Omni, KV across chunks, and turn-based paths: the answer

True incremental prefill or decode with KV kept across arriving input chunks:
- P2 `nemotron_duplex` conversation: yes (nano KV and Mamba state, talker KV, encoder window, per call).
- P3 `nemotron_duplex` over `/generate/ws`: yes, same state.
- P3 `qwen3_tts` live text: yes. Talker KV grows one decode step per text token; prefill once at open.
- P3 `waypoint` live actions: yes. Ring KV per world, one latent frame per action.
- Internal, not client input: Qwen3-Omni chunked Thinker prefill with the Talker extending its KV per chunk (`qwen3_omni_model.py:480-493`); Cosmos3 `kv` window mode across generated windows of one request.

Turn-based (re-prefill or re-run at commit or query, no KV across turns or steps):
- P1 Qwen3-ASR realtime transcription: each chunk is a whole-audio request.
- P5 Dynamo realtime (Qwen3-Omni): one chat request per commit.
- P6 Cosmos3 sessions across requests: latents re-encoded, not KV.
- P7 openpi, P8 windows and sentence groups: independent requests.

### F. Sessions: what the word means in each place

- Realtime transcription (`RealtimeSession`, `serving_realtime.py:174`): per WebSocket, not stored.
- Realtime conversation `session_id` (`sess_...`, `serving_realtime.py:997`): an id for events and request ids; nothing persists.
- Cosmos3 `SessionStore`: persists across requests (sections D-P6).
- Ring KV `session_idx` (`mstar/engine/resources/kv/ring/manager.py:169-181, 346-348`): one world slot per request id (Waypoint).
- Nemotron docstrings: "session" = one call = one engine request.
- `mstar/graph/base.py` has no session concept. Its streaming concepts are readiness for streamed inputs (`ReadySignals`, `:149-230`; `is_ready_for_streaming`, `:189-191`), and streamed inputs are not re-injected per loop iteration (`Loop.ingest_external_input` `:540-546`, `_is_streamed_input` `:550-552`). `NodeAndGraphWalk` (`:8-17`) names a node in a walk. **VERIFIED**.

### G. Infrastructure: `mstar/streaming/` (undescribed until now)

- `stream_buffer.py` (361 lines): `StreamBuffer` is per request, per edge, on the consumer worker (`:70-82`). Items arrive by `put` (`:177`) or inline by `put_values` (`:165`). `_update_buffer` (`:182-211`) joins chunked pieces along `join_dim` (`:212-232`). `has_chunk_ready` (`:264-285`) asks the policy; `pop_chunk` (`:287-350`) returns a `StreamChunk` with `chunk_index`, `start_offset`, `is_final`, `context_items`, `num_items`, and `released` (what the pop took out of the buffer, `:11-26`). `signal_done` (`:234-239`) and `_producer_done_and_all_read` (`:241-246`) end the stream. No state outlives a request.
- `chunk_policy.py` (340 lines): `ChunkPolicy` ABC (`:5-75`): `is_ready`, `next_chunk_size`, `window_size`, `prime`, `allow_partial_input`, `continue_after_producer_done`. Policies: `SlidingWindowChunkPolicy` (`:78`, Orpheus), `LeftContextChunkPolicy` (`:102`, vocoder warm-up), `FixedChunkPolicy` (`:148`, with `allow_partial` for per-chunk prefill at `:160-161`), `RampChunkPolicy` (`:194`, mel decoders), `ScheduledLeftContextChunkPolicy` (`:255`, Qwen3-TTS example at `:268-270`). **VERIFIED**.
- `topology.py` (250 lines): `StreamingGraphEdge` (`:15-27`); `CLIENT = "client"` pseudo-partition (`:34`); `ClientStreamLimits(max_items, item_size, unit)` (`:37-50`); `ClientAudio(sample_rate, frame_samples)` (`:70-77`); `Connection` with `join_dim`, `consumer_walk`, `audio`, `by_value` and its validation (`:80-121`); `PartitionTopology` (`:124-193`) incl. walk-driver checks (`:169-193`); `client_stream_limits` (`:196-207`); `find_client_stream` (`:231-250`). **VERIFIED**.
- Consumer side in `mstar/worker/stream_buffers.py` (438 lines): `consumer_nodes` (`:25`), `StreamBufferManager` (`:45`), `assign_consumer_walks` (`:144`), `pop_edge` (`:226`). **VERIFIED** (headers and signatures).
- Client intake in `mstar/api_server/data_worker.py`: `ClientCall` with `ended`, `charged` and `tails` (`:650-671`); `_charge_stream_items` (`:1406`) enforces `max_items`; `_client_audio_items` (`:1239`); `_framed` (`:1265`). **VERIFIED**.
- Deploy-time validation, `mstar/validation/client_streams.py` (135 lines): `check_client_streams` (`:20-42`); a `ClientAudio` stream needs its text and audio emitted from a loop, and every emitting partition must step once per item (`:84-97`); a client stream may not be sharded (`:30-39`); serving-config overrides may only lower `max_items` (`:107-135`). **VERIFIED**.
- Transport, `mstar/communication/communicator.py`: transport only, no stream or session logic. `OutboxCommunicator` (`:185-196`) queues sends a full peer cannot take, so a loop that reads its own input does not deadlock. `rust_communicator.py` is the same kind of transport. **VERIFIED**.
- `mstar/client/client.py`: one-shot SDK only (section D, P3). **VERIFIED**.

### H. (e) Audio and video: rates, alignment, timestamps

- No video input stream exists. The repo has exactly three client-stream connections: `nemotron_duplex` audio (`nemotron_duplex_model.py:499`), `qwen3_tts` text (`qwen3_tts_model.py:725`), `waypoint` actions (`waypoint_model.py:278`). Video enters only as a one-shot upload. **VERIFIED** (grep for `from_partition=CLIENT`).
- Audio: one client audio stream per call; the conversation route refuses more than one (`router.py:230-234`). Models with several client streams are validated to step once per item per stream (`validation/client_streams.py:84-97`), but none ships. Audio and video at different rates are therefore not supported for live input. **VERIFIED**.
- Rates: realtime routes accept `REALTIME_RATES` and resample on arrival (`serving_realtime.py:114`, `_put` `:1208-1234`). The engine rate is fixed per stream (16 kHz for Nemotron). Live file appends must match the stream rate (`data_worker.py:1257-1262`). **VERIFIED** (code); runtime **INFERRED**.
- Timestamps: none from the client. Server-side ones are the input sample clock (`serving_realtime.py:174-212`; `audio_start_ms`/`audio_end_ms` at `realtime_vad.py:287-300`), the wall-clock playback clock (`serving_realtime.py:1208-1253`), output `metadata["step"]` and `turn` (`data_worker.py:1392-1402`), `StreamChunk.start_offset` (internal, `stream_buffer.py:11-26`), and Waypoint's frame index in host state. **VERIFIED**.

## Could not verify

- No runtime run. Every "per step" and "KV kept" claim is from declared state, graph structure and code, not from an engine trace. In particular the KV retention claims for Nemotron, Waypoint, Qwen3-TTS and Cosmos3 `kv` mode need a run to confirm.
- P1: whether M*'s prefix cache hits on the Qwen3-ASR header before the audio across steps. **UNVERIFIED**.
- P1 Rust port: behaviour parity with `serving_realtime.py`. Not diffed. **UNVERIFIED**.
- P2 and P3 Nemotron: the exact encoder step per frame at runtime, and the output sample rate value (`eartts.sample_rate`, not read). **UNVERIFIED**.
- P3 Qwen3-TTS: Codec schedule and talker prefill internals beyond the topology and `text_in` path. Only partly read.
- P3 Waypoint: raw frames emitted per action (temporal compression), and ring sizing arithmetic (`waypoint_model.py:170-215`, `ring_geometry.py`). Not read in full. **UNVERIFIED**.
- P5 Dynamo: not executed; whether Dynamo deployments exist for Qwen3-Omni is not known.
- P6 Cosmos3: `kv` window pool release (`mstar/engine/windowing.py` and the KV manager) not traced past the header. Cosmos3 example client not run.
- Test evidence not read: `test/modular/test_openai_realtime.py`, `test/modular/test_client_stream_api.py`, `test/modular/test_stream_*.py`, `test/rust/test_rust_realtime*.py`, `test/modular/test_streaming_consumer_nodes.py`. These would show intended behaviour and were not checked.
- Conductor client-stream routing (`mstar/conductor/conductor.py:1514, 1615`) and data-worker route handling (`_route_client_streams`, `data_worker.py:1299`) were only grep-level.
- `mstar/model/components/fastconformer_stream.py` body (only its docstring) and `mstar/model/minicpm_o` audio encoder (block-causal attention, `minicpm_o_model.py:291-296`): no incremental use found in the graph, but the audio component was not read. **INFERRED** that it is not a client path.
- Whether any deployment config in `configs/` enables the Dynamo realtime surface or a realtime route for Qwen3-Omni was not checked beyond the repo grep.
