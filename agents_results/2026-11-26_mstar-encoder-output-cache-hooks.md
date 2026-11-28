# M* encoder-output cache: where it can hook into the graph, walk and engine machinery

## Question asked

Where could a generic, content-keyed cache of encoder node outputs (vision_encoder, audio_encoder, or any encoder-like node, not the Talker) be inserted in M*'s graph/walk machinery? How do walks decide which nodes run, so that an encoder can be skipped when its output is cached, or when the Thinker prefix cache already covers its span?

- Q1. Walk/node model, node dispatch, what a node receives and returns.
- Q2. Insertion points: (a) after a node finishes, before outputs go to the next node; (b) before dispatch, where a hit could bypass the node. Skip-drop versus replay. Whether a stored entry must carry deepstack.
- Q3. Prefix-skip interaction: can the vision node be dropped when the Thinker prefix covers its span? Exact condition, and what breaks on a partial hit.
- Q4. Are the encoder nodes in a shared component module, and which models share each encoder class?
- Q5. Size estimate for a minimal generic cache, files and functions, risks.

## Scope

- Repo: `/home/chaofan/workspaces/mm_reuse_study/external/mstar`, commit `2c8ffde8f`. Read only. No files under `external/` were modified. No servers, no GPU. No scratch files were written.
- Paths read: `mstar/graph/base.py`, `mstar/graph/runtime/rust.py`, `mstar/engine/engine.py`, `mstar/engine/resources/kv/manager.py`, `mstar/engine/resources/kv/config.py`, `mstar/worker/micro_scheduler.py`, `mstar/worker/executing_batch.py`, `mstar/worker/step_executor.py`, `mstar/worker/admission.py`, `mstar/conductor/conductor.py`, `mstar/conductor/request_info.py`, `mstar/model/base.py`, `mstar/model/submodule_base.py`, `mstar/model/multimodal.py`, `mstar/model/components/{block_loop_encoder,aut_submodule,aut_encoder}.py`, `mstar/model/qwen3_omni/{qwen3_omni_model,submodules,config}.py`, `mstar/model/qwen3_5/lora.py`, `mstar/model/cosmos3/{cosmos3_model,sessions}.py`, `mstar/model/bagel/bagel_model.py`, `mstar/api_server/data_worker.py`, `rust/src/graph/state.rs`.
- Checkpoint config read for vision sizes: `models/cache/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f79.../config.json`.
- Related notes checked for overlap: `agents_results/2026-11-25_qwen3-omni-g0-code-audit.md` (no encoder cache in M*), `2026-11-26_encoder-cache-capacity-streaming-frames.md` (vLLM-Omni ECM, not M*), `2026-11-26_mstar-qwen3-incremental-exactness.md` (M* TMRoPE and per-chunk encoder exactness).

Marks: **VERIFIED** = read in code. **INFERRED** = follows from code read, not executed. **UNVERIFIED** = not checked.

---

## Findings

### Q1. Walk/node model, dispatch, inputs and outputs

**Graph definition (VERIFIED).**
- A node is `GraphNode(name, input_names, outputs)`; a walk is a `GraphNode` or `Sequential([...])`; `Loop` wraps repeats. `mstar/graph/base.py:233-388` (GraphNode), `:390-443` (Sequential), `:485` (Loop).
- A node is ready when every input name has been ingested for its rid: `ReadySignals.update` `base.py:176-192`; the Rust twin is `rust/src/graph/state.rs:246-275` (`refresh_ready`: ready when `cur.mask == full_mask`) and `:318-340` (`ingest`). An edge with an empty `tensor_info` still counts as an ingested input (INFERRED from `ingest`, which does not look at the tensor count).
- The Qwen3-Omni prefill walks: `prefill_vision = Sequential([GraphNode("vision_encoder", inputs [pixel_values, image_grid_thw], outputs [vision_embeds, deepstack] -> Thinker), GraphNode("Thinker", inputs [vision_embeds, deepstack, video_second_per_grid, image_grid_thw, text_inputs])])`, `mstar/model/qwen3_omni/qwen3_omni_model.py:588-622`. The Thinker is ready only after the encoder's outputs arrive, so the encoder always completes before the Thinker's prefix probe.
- `node_groups` (`mstar/model/base.py:434-474`, `mstar/distributed/base.py`) is deployment placement of nodes on ranks. It does not select nodes per step.

**Who decides which walk runs (VERIFIED).**
- Plan per request, once, at admission: `conductor/conductor.py:1414-1452` (`pstate.plan = self.model.walk_plan(...)`, then validation of each `WalkStep`). `walk_plan` for the Thinker is `qwen3_omni_model.py:815-860`; the prefill steps come from `_build_thinker_prefill_schedule` `:887-905`, which uses `_thinker_steps` (`:208-221`). One `WalkStep` per attachment (text before it, plus text after for the last one), then `thinker_decode`.
- `WalkStep(walk, bind, step_metadata)` is `conductor/request_info.py:102-113`. `step_metadata` reaches every node of the walk, including `TEXT_BEFORE` and `FINAL_PREFILL` (`qwen3_omni_model.py:830-845`).
- The conductor enters a step with `_plan_step` (`conductor.py:2137-2152`): it builds one `GraphEdge` per walk input from persisted signals (or an empty edge). It advances with `_next_plan_args` (`:2154-2200`) when the worker reports. Neither function reads any cache state (VERIFIED). **The conductor cannot drop a step because it does not know the prefix hit.** The hit is decided later, in the worker, by the Thinker's own prepare.
- Worker scheduler: `worker/micro_scheduler.py:393` (`next_batch`) -> `_scan_ready` `:505-536` -> `runtime.get_ready_nodes` (`mstar/graph/runtime/rust.py:537-549`, Rust `state.rs` readiness) -> per-rid `check_ready` (`worker/admission.py:78`, which calls `Engine.check_ready` `engine/engine.py:3268-3306`). `check_ready` checks residency and resource admission (`admit_request`, reload, `admit_retrieve`). It does not probe the prefix cache. The batch is built in `worker/micro_scheduler.py:584` (`_schedule_key`) and `worker/batch_builder.py`.

**What a node receives (VERIFIED).**
- Inputs are tensor references (uuids) in the tensor store. `worker/executing_batch.py:122-130` resolves them with `services.streams.input_tensors` (`worker/stream_buffers.py:371`) into `per_request_input_tensors[rid]`, and builds `per_request_input_metadata` and `per_request_info` (`CurrentForwardPassInfo`: request_id, graph_walk, step_metadata, partition_name, fwd_index, random_seed; `worker/executing_batch.py:142-145`).

**Dispatch sequence on the GPU thread (VERIFIED).**
1. `worker/step_executor.py:126-226` (`__call__`): `engine.prepare_inputs(node_batch)` at `:189-197`, then `engine.exec_and_postprocess(node_batch)` at `:223`.
2. `engine/engine.py:1489-1585`: `prepare_inputs` / `_prepare_inputs`. Per rid, `submodule.prepare_inputs(...)` (`:1545-1557`), then `_skip_cached_prefix` for keyed walks (`:1571`), then `batch.register_prepare_batch(node_inputs)` (`:1583`) and `batch.drop_rids(skipped | failed)` (`:1585`).
3. `engine/engine.py:1893-1963` (`exec`): empty-batch shortcut at `:1909-1916`; `_exec_single` (`:1964`) or `_exec_per_request` (`:2074`) run declare, admit, plan, forward (`_drive_step`), then `_collect_outputs` (`:2796-2873`).
4. `engine/engine.py:2606-2633` (`exec_and_postprocess`): `postprocess_batch` (`:2543-2604`), which calls the submodule's per-rid `postprocess` hook (`mstar/model/submodule_base.py:1449-1482`); then `complete_cached_walk` for cached rids (`:2631-2632`).

**What a node returns (VERIFIED).**
- Per-rid dict of `NameToTensorList` (`BatchedModelOutput`). For the packed ViT these are per-request slices of the step's merged tensors (`mstar/model/components/block_loop_encoder.py:295-307`, `_split_rows`: slices, i.e. views).
- The worker stores outputs in the tensor store, minting new uuids: `worker/step_executor.py:1160-1255` (`_store_outputs`, calling `tensor_manager.store_and_return_tensor_info_batch`). Then the Rust runtime routes them: `rt.begin_postprocess` `:921`, `_send_input` `:1256`, and `rust/src/graph/state.rs:489-550` (`complete_strided`, which builds one `RoutedEdge` per output edge, `:142-150`).
- **Outputs are not stored per request.** They are handed to the next node as tensor references and dropped when the consumer dereferences them. The only per-iteration holding is for loops: `graph/base.py:614-630` (`Loop.maybe_cache_output`), which calls `tensor_manager.increment_ref` on each output uuid. This is an in-tree precedent for a worker holding output references.

### Q2. Insertion points

Two different things must be decided: **where** a hit is detected, and **what** the node-level result is.

**Skip-drop versus replay (VERIFIED by reading, consequences INFERRED).**
- **Plain skip (a `prepare_inputs` returning None)** does not complete the node. `submodule_base.py:1038-1052`: "None declines the row ... On any other step it is dropped." `step_executor.py:791` builds `valid_rids = request_ids | no_forward_rids`, so a dropped rid is discarded and is not routed. The node's consumers never receive an input for that rid. This is not a usable cache-hit path (INFERRED: the request would stall at the consumer; the stall behaviour was not traced).
- **Cached-completion skip (the existing `_skip_cached_prefix` pattern)** returns None *and* records the rid in `batch.cached_rids` (`engine/engine.py:1873`). `no_forward_rids` (`:591-596`) keeps it in the valid set, so the node completes and the rid is routed. But the outputs are empty (`outputs.get(rid)` is None), so the consumer gets zero-tensor edges.
- **Zero-tensor edges break the Thinker.** The Thinker's prefill_vision prepare reads `inputs["vision_embeds"][0]` unconditionally (`qwen3_omni/submodules.py:413`), and `get_input_sequence_len` reads `inputs[name][0].shape[0]` for `vision_embeds` (`submodules.py:358-362`, `_EMBED_INPUTS` at `:348`). The scheduler measures the Thinker's length before prepare. So an empty vision bundle raises (INFERRED: a per-request failure, via `register_failure` at `engine.py:1560-1582`).
- **Therefore a hit that skips the encoder must replay its outputs**, that is, put the cached tensors back into the node's outputs so the store and route run as normal. This is the recommended design.

**(a) Write point: after the node finishes, before its outputs are handed on.**
- `engine/engine.py:2606-2627` (`exec_and_postprocess`), after `postprocess_batch`: the per-rid outputs are final and the step has not yet been stored. Recommended.
- Alternatively `engine/engine.py:2562-2587` (`_postprocess_batch`, the per-rid `submodule.postprocess` loop), or the model hook `mstar/model/submodule_base.py:1449-1482` (`postprocess(request_id, request_info, outputs, inputs)`), which is per-model. The hook is the least intrusive place for a model to opt in.
- Keys available there: `batch.node_name` (engine.py:2562 `_postprocess_batch`), the rid's walk `batch.step_context.walk_of(rid)`, `batch.per_request_info_wrapped[rid]` (`CurrentForwardPassInfo`, including `step_metadata`), `batch.per_request_input_tensors[rid]` (which holds the raw `image_grid_thw` / `video_grid_thw` tensors, `executing_batch.py:122-130`), and the submodule's model fingerprint. **The media content digest is not in any of these today.** It must be delivered (see below).
- Outputs must be **cloned** before caching: they are views of the step's merged tensors (`block_loop_encoder.py:295-307`).

**(b) Read point: before dispatch, where a hit could bypass the node.**
- `engine/engine.py:1555-1580` (`_prepare_inputs` per-rid loop), next to the existing `_skip_cached_prefix` call at `:1571`. A new `_serve_cached_output(batch, rid, inputs)` check goes here: on a hit, record the rid in a new `batch.served_rids`, add it to `no_forward_rids` (`:591-596`), keep it out of `request_ids` (`drop_rids`, `:629-650`), and return None.
- The empty-batch shortcut (`engine.py:1909-1916`) must still produce entries for served rids. The easiest place is the batch that is built in `exec_and_postprocess`.
- `worker/micro_scheduler.py:505-536` (`_scan_ready`) and `engine/engine.py:3268` (`check_ready`) are **not** recommended. They gate readiness, but they have no outputs path. A replay needs the outputs to be set after the forward is skipped.

**Replay point.**
- `engine/engine.py:2606-2633` (`exec_and_postprocess`): before `postprocess_batch`, set `outputs.per_rid_outputs[rid] = cached_outputs` for each served rid. Then `step_executor._store_outputs` (`:1160`) mints fresh uuids for them, and routing proceeds as for a fresh row. `valid_rids` already includes `no_forward_rids` (`step_executor.py:791`), so nothing else changes.
- The downstream Thinker is unchanged: its prepare still sees the same tensor names. This is the property that makes replay safe for the Thinker's own prefix probe (Q3).

**Skip versus replay, summary.**
- Skip (drop the node, no outputs): not usable, because downstream reads the bundle unconditionally (above).
- Replay (cached tensors re-enter the normal store and route): the only option without changing the Thinker.

**Deepstack (VERIFIED).**
- The Qwen3-Omni vision node emits **two** edges: `vision_embeds` and `deepstack` (`qwen3_omni_model.py:590-600`). `deepstack` is a list with one tensor per deepstack block (`qwen3_omni/submodules.py:240-258`, `forward_batched`: "one deepstack tensor of that size per deepstack block"). Each is `[merged_tokens, out_hidden]`.
- The Thinker reads the list (`submodules.py:455-468`, `inputs["deepstack"]`), and cuts it for chunks (`submodules.py:578-586`, `cut.tensor_inputs["deepstack"]`).
- Checkpoint config (read): `out_hidden_size` 2048, `deepstack_visual_indexes` [8, 16, 24] (3 blocks). So the stored entry **must carry all four tensors** (vision_embeds + 3 deepstack), or the Thinker's prefill_vision prepare fails or produces a wrong residual.
- `rebuild_policies` already treats the bundle as one unit: `copy_width` = `out_hidden * (1 + len(deepstack_visual_indexes))` (`qwen3_omni_model.py:495-507`).
- Size: 11520 merged rows (the video example in the prior audit) x 2048 x 4 tensors x 2 bytes (bf16 assumed, INFERRED) = **about 189 MB per video**.

**Where the digest must come from.**
- `_digest` (`api_server/data_worker.py:1605-1630`) hashes the whole file with `_sha256` (`:135-137`), checks `_file_state` against the state at intake (`:1623-1626`), and folds in `preprocess_fingerprint()`, `span.advance` and `span.params`. It is called **only** for keyed spans (`data_worker.py:1494-1499`). A video prompt has `keyed_spans = 1` (`qwen3_omni_model.py:1199-1201`), so today no digest is computed for the video. An encoder cache would need digests for all media spans, keyed or not (INFERRED design requirement).
- Route into the worker: `walk_plan(..., model_kwargs)` already receives per-request kwargs (`qwen3_omni_model.py:815-821`). A digest passed through model_kwargs (as the prefix keys already are, read at `conductor/conductor.py:1463`) can be placed in each walk step's `step_metadata` and reach the vision node as `fwd_info.step_metadata` (INFERRED; the prefix-key path is the precedent).

### Q3. Prefix-skip interaction

**Walk and probe order (VERIFIED).**
- The Thinker's probe for a keyed walk runs in `_prepare_inputs` (`engine/engine.py:1571`, `_skip_cached_prefix` at `:1847-1879`), or on the first chunk in `_prepare_chunk` (`:1675-1686`, `resolve_cached_prefix`). Both run after the vision node has completed, because the Thinker is not ready until `vision_embeds` arrives (`graph/base.py`/`state.rs` readiness; see Q1).
- So the vision node always runs before the Thinker's probe. Nothing in the walk plan, the conductor or the scheduler can drop it on a prefix hit (VERIFIED).

**Keying for a video prompt (VERIFIED).**
- `_state_layout` sets `metadata["keyed_spans"] = {THINKER_KV: {"main": 1}}` whenever a video is in the plan, with the comment "keying past it has never been run or tested" (`qwen3_omni_model.py:1199-1201`). Only the first span, the leading text, is keyed.
- `keyed_count` returns `keyed_spans` when set, else all spans up to the first ROWS span (`mstar/model/base.py:228-231`). For images, `keyed_spans` is unset, so the image digest span is keyed (`base.py:210-225`, default None).
- The lookup keys only the pages of the keyed chain, and drops the last one: `walk(rooted)[:len(rooted) - 1]` (`engine/resources/kv/manager.py:1036`). The probe answer is `_answer` (`manager.py:1201-1215`): `tokens = min(max(present - cursor.consumed, 0), length)`. A partly held item does not move the position ("A partly held item doesn't move the position. Its walk still runs and advances it.", `manager.py:1212`).

**Exact skip condition (VERIFIED condition; the consequences are the reasoning in the next bullets).** Let `h` be the tokens the Thinker's probe reports for the `prefill_vision` step (`resolve_cached_prefix`, `manager.py:976-1019`), counted from the start of that step's input. Let `[v0, v1)` be the media span's KV slots within that step (for video: the vision rows plus the two sentinels; for an image: `image_span`). The vision node's outputs can be skipped without changing the Thinker's output if and only if:
1. `h >= v1` (every KV slot the media writes is already held, so the Thinker writes no media row and reads no media embedding or deepstack value), **and**
2. the Thinker still receives its non-encoder inputs (`image_grid_thw` / `video_grid_thw`, `video_second_per_grid`, text) and the `vision_embeds` **shape**, which `get_input_sequence_len` reads (`submodules.py:358-362`). Items 1 and 2 are satisfied by a replay, or by a change in the Thinker's prepare so that it does not read `vision_embeds` for a fully held row.

**Consequences for the current code.**
- **Video prompts: never.** The keyed chain holds only leading-text pages, so `h <= len(leading text) < v0 <= v1`. The condition fails for every hit, so the vision node is never droppable on a video prompt under the current keying, regardless of cache state (VERIFIED from the keying and `walk(rooted)[:len-1]`; INFERRED that `h` cannot cross `v0`).
- **Image prompts: possible.** The image span is keyed (`base.py:228-231`), so `h >= v1` is reachable when the image's KV is held and at least one more page of text remains after it (INFERRED from `walk(rooted)[:len-1]` and `_image_end`, `manager.py:1063-1070`). This is the BAGEL-style case in the prior audit: the ViT runs and its output is unused. The same Sequential shape holds for BAGEL (`bagel_model.py:1258-1280`, `prefill_vit` = `vit_encoder` -> `LLM`).

**What breaks on a partial hit (h < v1, e.g. only the leading text is held).**
- The Thinker must prefill media rows `h..v1` and needs `vision_embeds` and `deepstack` for them (`submodules.py:413-468`, splicing `wrapped_embeds` and `deepstack` into the row, and `full_deepstack[start:start + vision_len]`).
- Without the vision node: the prepare reads `inputs["vision_embeds"][0]` on an empty list (`submodules.py:413`), so the request fails. If someone "fixed" that by zero-filling, the Thinker would write wrong KV for the video rows and the text would be silently wrong.
- With a replay: the full bundle must be replayed, not a slice. The chunked Thinker cuts the bundle per chunk (`engine.py:1641-1706`, `submodule.split_inputs`), and the first chunk starts past `h`. So a partial hit is safe **only** with a full-bundle replay.

### Q4. Generic component or model-specific

**Encoder node classes (VERIFIED).**

| Shared component (module) | Encoder class | Models that use it |
|---|---|---|
| `PackedBlockLoopEncoder` (`mstar/model/components/block_loop_encoder.py:67`, `MAX_BATCH_SIZE = 1` at `:74`) | per-model subclass | Qwen3-Omni vision (`qwen3_omni/submodules.py:156`); Qwen3.5 vision (`qwen3_5/submodules.py:245`, LoRA root on `vision_encoder` at `qwen3_5/lora.py:55-66`); Cosmos3 vision (`cosmos3/submodules.py:3285`); Muse vision (`muse/submodules.py:43`); Phi4MM vision (`phi4mm/vision.py:295`) |
| `AuTEncoderSubmodule` (`mstar/model/components/aut_submodule.py:34`, `MAX_BATCH_SIZE = 8` at `:40`) and `AuTEncoder` (`components/aut_encoder.py:214`) | per-model subclass | Qwen3-Omni audio (`qwen3_omni/submodules.py:128`); Qwen3-ASR audio (`qwen3_asr/submodules.py:36`); `AuTEncoderLayer` is also used by MiniCPM-o audio (`minicpm_o/components/audio.py:19`) |

**Encoder classes that are model-specific (VERIFIED).**
- BAGEL `ViTEncoderSubmodule` (`bagel/submodules.py:361`), node `vit_encoder` (`bagel_model.py:608-617`).
- MiniCPM-o vision and audio (`minicpm_o/submodules.py:304`, `:403`), nodes `vision_encoder` / `audio_encoder` (`minicpm_o_model.py:115-116`).
- Whisper (`whisper/submodules.py:96`); Phi4MM audio (`phi4mm/audio.py:183`); Command A+ vision (`command_a_plus/components/vision.py:112`, node `vision_encoder`).

**Shared node names (VERIFIED).**
- `vision_encoder`: Qwen3-Omni, Qwen3.5, Cosmos3 (`cosmos3_model.py:161`), Phi4MM (`phi4mm_model.py:228`), MiniCPM-o, Muse, Command A+.
- `audio_encoder`: Qwen3-Omni, Qwen3-ASR (`qwen3_asr/config.py:28`), Phi4MM (`phi4mm_model.py:234`), MiniCPM-o.

**Shared descriptor (VERIFIED).** `MediaEncoder(name, node, inputs, output)` in `mstar/model/multimodal.py:440-461` (with `vision_encoder()` at `:483-485`) is data only. It is used by Qwen3-ASR, Cosmos3, canary_qwen and the shared `MediaIntake` path. **Qwen3-Omni does not use it**: it has its own `_MM_WALKS` table (`qwen3_omni_model.py:872-883`).

**Implication for placement (INFERRED).** A cache at the component level (`PackedBlockLoopEncoder` / `AuTEncoderSubmodule`) covers Qwen3-Omni vision and audio, Qwen3-ASR audio, and the vision towers of Qwen3.5, Cosmos3, Muse and Phi4MM, but not BAGEL, MiniCPM-o, Whisper or Command A+. A cache at the **engine/graph** layer (Q2 points) covers any node that opts in through a per-submodule hook. The engine layer is the better place, with the component classes only supplying a default key.

**Caveats that are model-specific (VERIFIED).**
- Qwen3.5 has a LoRA root on the vision node (`qwen3_5/lora.py:55-66`), so its vision outputs depend on the adapter. The key must include the adapter.
- Qwen3-Omni refuses LoRA on the encoders (`qwen3_omni_model.py:350-362`, docstring "one touching the Talker or the encoders ... is refused"). Its key needs no adapter id.
- Cosmos3 declares no prefix keys at all (`cosmos3_model.py:486-493`, `kv_labels` has no `keys=`; `base.py:333-339` keeps only labels with keys). Its vision node has no prefix-hit path to share with.

### Q5. Size estimate for a minimal generic cache

**Files and functions (estimates, INFERRED).**

| Where | What | Approx. size |
|---|---|---|
| New: `mstar/engine/resources/encoder_cache.py` (or `mstar/model/components/`) | bounded store keyed by a fingerprint; LRU; byte budget; pins for entries in use; lock. Template: `mstar/model/cosmos3/sessions.py` (`SessionStore`, `begin` pins entries, `:50-75`) | 150-200 lines |
| `mstar/model/submodule_base.py` | per-submodule opt-in hook, e.g. `output_cache_key(graph_walk, fwd_info, inputs) -> bytes | None` (default None), next to `postprocess` (`:1449`) | 15-25 lines |
| `mstar/engine/engine.py` | `_prepare_inputs` serve check beside `:1571`; `ExecutingBatch.served_rids` (`:498-501`), `no_forward_rids` (`:591`), `drop_rids` (`:629`); empty-batch branch (`:1909`); fill outputs and write in `exec_and_postprocess` (`:2606-2633`) | 60-80 lines |
| `mstar/engine/engine.py` (TP) | rank 0 decides, followers adopt, as `decide_cached_prefix` / `adopt_cached_prefix` (`:1779-1833`) | 40-60 lines, only if the encoder node is TP-sharded |
| `mstar/api_server/data_worker.py` | compute digests for all media spans (not only keyed ones), reusing `_digest` (`:1605-1630`) and `_file_state` (`:1623`); pass through model_kwargs; per-request opt-out via passthrough | 40-60 lines |
| `mstar/model/qwen3_omni/qwen3_omni_model.py` | `walk_plan` puts digests into `step_metadata` (`:815-860`); `preprocess_fingerprint` override with checkpoint and vision config (`base.py:874-881`, currently class name only) | 20-30 lines |
| `mstar/model/qwen3_omni/submodules.py` | `VisionEncoderSubmodule` / `AudioEncoderSubmodule` declare outputs and clone for the cache (`:128`, `:156`) | 15-25 lines |
| Tests | model on the existing prefix-cache test as template (`test/modular/test_qwen3_omni_prefix_cache.py`, referenced in the prior audit) | 150-250 lines |

Total: roughly 400-500 lines of change across 8-9 files, plus tests. (Estimate, not measured.)

**Risks.**
1. **Memory placement.** A video's bundle is about 189 MB (Q2). On the 48 GB cards the Thinker already takes most of the memory at TP2, so a device-resident cache competes with Thinker KV. A host (pinned) store with a byte cap is safer, at the cost of a D2H copy on store and an H2D copy on hit (about 190 MB each way for a video; timing not measured here, UNVERIFIED). Default should be small or off.
2. **Key completeness.**
   - File content: `_sha256` (`data_worker.py:135-137`) and the `_file_state` guard (`:1623-1626`) are reusable.
   - Model identity: `preprocess_fingerprint` returns only the class name (`base.py:874-881`). Two checkpoints of one class would share keys. Must add the checkpoint identity and vision config (dtype included).
   - Preprocessing parameters: `load_video` (`qwen3_omni_model.py:916-940`) and `process_prompt`'s resize (`:1109-1124`) are fixed in code, so they are constants for the key (a version constant is enough). `VideoBudget` (`mstar/model/media_limits.py:552-571`) raises rather than changing the output, so it does not need a key. The existing KV digest uses `params = ()` for video, which is the same gap if these become configurable.
   - LoRA: Qwen3.5's vision outputs depend on the adapter (Q4). The key must include the adapter id there.
3. **Invalidation.** Keys must change on any weight, config, or preprocessing change (a fingerprint over those), with an LRU bound and optional TTL.
4. **Thread safety.** Prepare, exec and postprocess run on the GPU thread (`step_executor.py:126-226`). Eviction or `cleanup_request` (`submodule_base.py:1578`) may run elsewhere, so the store needs a lock. In-flight entries must be pinned (the `SessionStore` pattern).
5. **Numerical identity.** A hit returns the stored bundle. It is bitwise identical to a fresh run only if the encoder is batch-invariant. `PackedBlockLoopEncoder` runs one request per step (`MAX_BATCH_SIZE = 1`, `block_loop_encoder.py:74`), which helps. `AuTEncoderSubmodule` batches up to 8 (`aut_submodule.py:40`), so audio outputs may depend on batch composition. Needs a test (INFERRED).
6. **Replicas and TP.** The cache lives in the encoder's worker. Hit rate depends on the conductor sending repeat requests to the same replica (replica scoring not traced). If the encoder is TP-sharded, the hit decision must be made on rank 0 and sent (precedent in `decide_cached_prefix` / `adopt_cached_prefix`, `engine.py:1779-1833`). In the 3-GPU M* config the encoders sit on rank 0 (prior audit, not re-read here).
7. **Audio output.** The conductor turns off the prefix cache when a streamed partition runs (`conductor/conductor.py:1466-1468`). The encoder cache is independent of prefix keying, so it should have its own switch, and should keep working for audio-output requests.
8. **Gain bound.** For a video, the Thinker's prefill of about 11.5k tokens is probably the larger cost, so the encoder cache saves only the ViT share. The ViT share was not measured in this repo (UNVERIFIED).

---

## Insertion-point table

| # | Point | file:line | Key available | Use |
|---|---|---|---|---|
| 1 | Before dispatch, per rid (read / serve check) | `mstar/engine/engine.py:1555-1580` (next to `_skip_cached_prefix` at `:1571`) | node name (`batch.node_name`), walk (`step_context.walk_of`), `fwd_info.step_metadata` (digest, once delivered), row inputs | Hit: mark served, no forward, keep in `no_forward_rids` (`:591-596`) |
| 2 | Empty-batch shortcut (must still yield served rids) | `mstar/engine/engine.py:1909-1916` | as above | Served rids carried to postprocess |
| 3 | After forward, before store (replay) | `mstar/engine/engine.py:2606-2633` (`exec_and_postprocess`) | served rids and their cached bundle | Set `outputs.per_rid_outputs[rid]`; the store then mints new uuids |
| 4 | After node finishes (write) | `mstar/engine/engine.py:2562-2604` / `2606-2627`; model hook `mstar/model/submodule_base.py:1449-1482` | node, walk, rid, `fwd_info`, outputs (clone them) | Insert the bundle into the cache |
| 5 | Store and route (unchanged) | `mstar/worker/step_executor.py:1160-1255` (`_store_outputs`), `:791` (`valid_rids`), `:1256` (`_send_input`); Rust `rust/src/graph/state.rs:489-550` | outputs | No change needed |
| 6 | Digest delivery | `mstar/api_server/data_worker.py:1471-1500` (`_prefix_keys`, digests for keyed spans only), `:1605-1630` (`_digest`); `mstar/model/qwen3_omni/qwen3_omni_model.py:815-860` (`walk_plan`, step_metadata) | file sha256, preprocess fingerprint, span advance and params | Key component |
| 7 | Plan-level skip (not recommended) | `mstar/conductor/conductor.py:1414-1452` (plan), `:2137-2200` (step entry) | none of the cache state | Conductor does not know hits |
| 8 | Scheduler readiness (not recommended) | `mstar/worker/micro_scheduler.py:505-536`; `mstar/engine/engine.py:3268-3306` | none of the outputs | Gates readiness only |
| 9 | Thinker probe (existing, unchanged) | `mstar/engine/engine.py:1847-1879` (`_skip_cached_prefix`), `:1641-1706` (first chunk) | KV layout | Runs after the vision node; cannot skip it |

---

## Could not verify

- **Runtime behaviour**: no GPU run, no server, no test. In particular, the exact stall or failure of a consumer when a node is dropped by a plain `None` from `prepare_inputs` (not traced through the Rust runtime).
- **Whether the vision step is `rows_only`**: INFERRED that the vision node's outputs are per-rid dicts (`engine.py:1612`, `:2621`; `device_loopback_signals` default empty, `submodule_base.py:1237`), not checked at runtime.
- **Numerical identity** of a cached bundle versus a fresh run (batch composition, capture buckets, ragged attention padding). Needs a test.
- **ViT share of time** for a video prefill. No M* timing for the vision node was found in the repo (search of `agents_results/` and `results/` found none).
- **Replica routing**: whether the conductor's replica pick (`conductor.py`, `pick` scoring) would send a repeat video to the same replica.
- **TP behaviour** of the encoder node in the deployed configs (only the prior audit's statement that encoders sit on rank 0 was used).
- **dtype** of the stored bundle: bf16 assumed for the 189 MB estimate.
- **Qwen3-ASR and BAGEL**: only the graph shape and node names were read; their prefix-hit paths were not traced in detail.
- **Thinker output on a zero-filled or missing bundle**: not run; the failure mode is inferred from the prepare code.
