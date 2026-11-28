"""把 external/mstar 的本研究改动拆成按顺序叠加的系列补丁（patches/mstar-series-<tag>/NN-<编号>.patch）。

从当前工作树（全部改动）出发，按倒序逐项剥离：诊断 → C4 → C2 → C1 → M4 → M3b → M3 → M1 → M0。
每一步都是精确替换（每处必须恰好命中一次），并检查 Python 能编译。全部剥完必须与上游 HEAD 逐字节相同，
否则说明有改动没归到任何一项。然后在临时 git worktree 里按正序提交每个快照，format-patch 导出。

python scripts/patches/split_mstar_series.py [--out <目录>]
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external" / "mstar"
FILES = [
    "mstar/api_server/data_worker.py",
    "mstar/conductor/conductor.py",
    "mstar/engine/engine.py",
    "mstar/engine/resources/kv/manager.py",
    "mstar/model/bagel/submodules.py",
    "mstar/model/qwen3_omni/qwen3_omni_model.py",
    "mstar/model/qwen3_omni/submodules.py",
    "mstar/model/components/encoder_output_cache.py",
    "mstar/model/components/c1_registry.py",
]
MODEL = "mstar/model/qwen3_omni/qwen3_omni_model.py"
SUB = "mstar/model/qwen3_omni/submodules.py"
ENG = "mstar/engine/engine.py"
CACHE = "mstar/model/components/encoder_output_cache.py"

# (序号, 编号, 标题) —— 正序
STAGES = [
    ("00", "M0", "Qwen3-Omni Talker/Code2Wav 可不建 CUDA graph（MSTAR_SPEECH_NODES_EAGER）"),
    ("01", "M1", "编码器输出缓存：Qwen3-Omni 视觉编码器与 BAGEL ViT（MSTAR_ENCODER_CACHE_MB）"),
    ("02", "M3", "相邻图像合成一个 walk（MSTAR_GROUP_IMAGES）"),
    ("03", "M3b", "图像预处理缓存与 data worker 计时（MSTAR_PREPROCESS_CACHE_MB / MSTAR_PREPROCESS_LOG）"),
    ("04", "M4", "媒体键随图发送、bf16 像素、不存解码图、文件记忆（MSTAR_MEDIA_KEYS）"),
    ("05", "C1", "前缀租约已覆盖的图不编码、不传（MSTAR_C1）"),
    ("06", "C2", "编码提前：VisionAhead 分区与第二个 ViT 节点（MSTAR_C2）"),
    ("07", "C4", "摄像头会话：已上传的帧发引用、预热组整组跳过（MSTAR_C4）"),
    ("08", "diag", "诊断日志：Thinker prefill 步计时（MSTAR_FWD_LOG）、C1 登记（MSTAR_C1_LOG）"),
]


class Tree:
    def __init__(self, files: dict[str, str | None]):
        self.f = dict(files)

    def r(self, path: str, old: str, new: str = "") -> None:
        text = self.f[path]
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: 期望命中 1 次，实际 {n} 次：{old[:120]!r}")
        self.f[path] = text.replace(old, new)

    def cut(self, path: str, start: str, end: str, keep_end: bool = True, new: str = "") -> None:
        """删去从 ``start`` 起到 ``end`` 之前（keep_end）或之后的文本。"""
        text = self.f[path]
        if text.count(start) != 1:
            raise SystemExit(f"{path}: 起点命中 {text.count(start)} 次：{start[:120]!r}")
        i = text.index(start)
        j = text.index(end, i + len(start))
        j = j if keep_end else j + len(end)
        self.f[path] = text[:i] + new + text[j:]

    def snapshot(self) -> dict[str, str | None]:
        return dict(self.f)


# ---------------------------------------------------------------------------- 各项的剥离（从后往前）

def strip_diag(t: Tree) -> None:
    t.r(ENG, '_FWD_LOG = os.environ.get("MSTAR_FWD_LOG")   # mm_reuse_study diagnostic\n')
    t.cut(ENG, '        if _FWD_LOG and batch.node_name == "Thinker"',
          '        if lease is not None:\n            ctx = batch.step_context\n            if ctx.resident',
          new='')
    t.r(ENG, '''            if os.environ.get("MSTAR_C1_LOG"):
                with open(os.environ["MSTAR_C1_LOG"], "a") as f:
                    f.write(f"{time.time():.6f}\\t{self._graph_runtime.get_rid_string(rid)}\\tpublish\\t{walk}\\t"
                            f"key={key}\\tcovered={n}\\n")
''')


def strip_c4(t: Tree) -> None:
    dw = "mstar/api_server/data_worker.py"
    t.cut(dw, '_C4_FRAMES_DIR = os.environ.get("MSTAR_C4_FRAMES_DIR"', "\nlogger = logging.getLogger", new="")
    t.r(dw, "raw = _sha256(_c4_original(paths[index]))", "raw = _sha256(paths[index])")

    # 分组与步计划
    t.r(MODEL, "def _thinker_steps(plan: list[PromptPart], warm: int = 0)", "def _thinker_steps(plan: list[PromptPart])")
    t.r(MODEL, "for group in _media_groups(plan, media, warm):", "for group in _media_groups(plan, media):")
    t.cut(MODEL, "# mm_reuse_study C4 (MSTAR_C4=1, needs C1)", 'VISION_AHEAD = "vision_ahead"', new="")
    t.cut(MODEL, "def _c4_warm(model_kwargs)", "def _key_tensor", new="")
    t.r(MODEL, '''def _media_groups(plan: list[PromptPart], media: list[int], warm: int = 0) -> list[list[int]]:
    """The attachments, each its own group, or with ``GROUP_IMAGES`` each run of
    adjacent images (plan neighbours, so no text between them) as one group.
    C4: the first ``warm`` images (adjacent, from the first attachment) are one group of their own."""
    groups: list[list[int]] = []
    if warm:
        lead = media[:warm]
        if len(lead) != warm or any(plan[i].modality != "image" for i in lead) or lead != list(range(lead[0], lead[0] + warm)):
            raise ValueError(f"c4_warm={warm} must count adjacent leading images of the prompt")
        groups.append(lead)
        media = media[warm:]
''', '''def _media_groups(plan: list[PromptPart], media: list[int]) -> list[list[int]]:
    """The attachments, each its own group, or with ``GROUP_IMAGES`` each run of
    adjacent images (plan neighbours, so no text between them) as one group."""
    groups: list[list[int]] = []
''')
    t.r(MODEL, "GROUP_IMAGES > 1 and groups and len(groups) > (1 if warm else 0) and len(groups[-1])",
        "GROUP_IMAGES > 1 and groups and len(groups[-1])")
    t.r(MODEL, "            | ({C4_WARM} if C4 else set())\n")
    t.r(MODEL, '''            warm = _c4_warm(model_kwargs)
            prefills = self._build_thinker_prefill_schedule(input_modalities, warm)
''', "            prefills = self._build_thinker_prefill_schedule(input_modalities)\n")
    t.r(MODEL, "                steps = _thinker_steps(prefill_plan(parts_from_modalities(input_modalities)), warm)\n",
        "                steps = _thinker_steps(prefill_plan(parts_from_modalities(input_modalities)))\n")
    t.r(MODEL, "self._vision_ahead_plan(input_modalities, audio_output, _c4_warm(model_kwargs))",
        "self._vision_ahead_plan(input_modalities, audio_output)")
    t.r(MODEL, "audio_output: bool, warm: int = 0) -> list[WalkStep]:", "audio_output: bool) -> list[WalkStep]:")
    t.r(MODEL, "        plan = []\n        steps = _thinker_steps(prefill_plan(parts_from_modalities(input_modalities)), warm)\n",
        "        plan = []\n        steps = _thinker_steps(prefill_plan(parts_from_modalities(input_modalities)))\n")
    t.r(MODEL, '''        if warm:   # C4: the session's warm group is held (C1 skips it) and travels without pixels
            groups = [g for g in groups if g][1:]
''')
    t.r(MODEL, '''    def _build_thinker_prefill_schedule(
        self, input_modalities: list[str], warm: int = 0,
    ) -> list[tuple[str, dict[str, str], int]]:''',
        "    def _build_thinker_prefill_schedule(self, input_modalities: list[str]) -> list[tuple[str, dict[str, str], int]]:")
    t.r(MODEL, "enumerate(_thinker_steps(prefill_plan(parts_from_modalities(input_modalities)), warm)):",
        "enumerate(_thinker_steps(prefill_plan(parts_from_modalities(input_modalities)))):")

    # 帧引用
    t.r(MODEL, '''        placeholder carrying that output (``_m4``), without decoding.
        C4: a frame reference (a file holding ``MSTAR-FRAME:<xxh3-128 of the frame's bytes>``, sent as a tiny
        data URL) names a frame the server saved before (``C4_FRAMES_DIR``)."""''',
        '''        placeholder carrying that output (``_m4``), without decoding."""''')
    t.cut(MODEL, "            raw = f.read()\n        if C4 and raw.startswith(C4_MAGIC):",
          "        digest = xxhash.xxh3_128_hexdigest(raw)\n", keep_end=False,
          new="            digest = xxhash.xxh3_128_hexdigest(f.read())\n")
    t.cut(MODEL, "        if C4:   # keep the frame's bytes for later references", "        return out\n", new="")

    # 预热组不带像素
    t.cut(MODEL, "        warm = _c4_warm(kwargs)\n", "        if not attached:\n", new="")
    t.r(MODEL, '''            return ProcessPromptOutput(self._split_steps(result, plan, warm), {})
        metadata = self._state_layout(plan, result, "audio" in output_modalities, warm)
        return ProcessPromptOutput(self._split_steps(result, plan, warm), metadata)''',
        '''            return ProcessPromptOutput(self._split_steps(result, plan), {})
        metadata = self._state_layout(plan, result, "audio" in output_modalities)
        return ProcessPromptOutput(self._split_steps(result, plan), metadata)''')
    t.r(MODEL, "plan: list[PromptPart], warm: int = 0) -> NameToTensorList:", "plan: list[PromptPart]) -> NameToTensorList:")
    t.r(MODEL, "        steps = _thinker_steps(plan, warm)\n", "        steps = _thinker_steps(plan)\n")
    t.r(MODEL, '''audio_output: bool,
                      warm: int = 0) -> dict:''', "audio_output: bool) -> dict:")
    t.r(MODEL, "for k, step in enumerate(_thinker_steps(plan, warm)):", "for k, step in enumerate(_thinker_steps(plan)):")

    t.cut(SUB, "        # mm_reuse_study C4: a session's warm group travels without", "        if skip:   # mm_reuse_study C1", new="")
    t.r(SUB, "pixel_values[sum(t * h * w for t, h, w in grid[absent:skip]):]",
        "pixel_values[sum(t * h * w for t, h, w in grid[:skip]):]")


def strip_c2(t: Tree) -> None:
    t.cut(MODEL, "# mm_reuse_study C2 (MSTAR_C2=1, needs M1)", '\n\ndef _key_tensor', new="")
    t.r(MODEL, 'nodes={"vision_encoder", *([VISION_AHEAD] if C2 else [])}', 'nodes={"vision_encoder"}')
    t.cut(MODEL, "            **({VISION_AHEAD: GraphNode(", "        }\n", new="")
    t.r(MODEL, '            *([PartitionDefinition(name="VisionAhead", graph_walks={VISION_AHEAD})] if C2 else []),\n')
    t.r(MODEL, 'partitions=["Thinker", "Talker", "Code2Wav", *(["VisionAhead"] if C2 else [])],',
        'partitions=["Thinker", "Talker", "Code2Wav"],')
    t.r(MODEL, '        if partition_name == "VisionAhead":\n'
               '            return self._vision_ahead_plan(input_modalities, audio_output)\n')
    t.cut(MODEL, "    def _vision_ahead_plan(", "    @staticmethod\n    def _decode_walk", new="")
    t.r(MODEL, 'elif node_name in ("vision_encoder", VISION_AHEAD):', 'elif node_name == "vision_encoder":')

    t.r(SUB, "from mstar.model.qwen3_omni.qwen3_omni_model import C2, VISION_AHEAD  # mm_reuse_study C2\n")
    t.r(SUB, "_C2_CACHE = None   # mm_reuse_study C2: the first ViT node's output cache in this process\n\n\n")
    t.cut(SUB, "        if C2 and self.output_cache.enabled:", "\n    def bind_state", new="")
    t.r(SUB, "        ahead = graph_walk == VISION_AHEAD\n        skip = self._c1_skip(fwd_info, len(grid), drop=not ahead)\n",
        "        skip = self._c1_skip(fwd_info, len(grid))\n")
    t.cut(SUB, "        if ahead and self.output_cache.enabled:", "        if self.output_cache.enabled:\n", new="")
    t.r(SUB, "            self.output_cache.device = device   # C2: the other node may have filled the cache\n")
    t.r(SUB, "items: int, drop: bool = True) -> int:", "items: int) -> int:")
    t.r(SUB, "        if skip < items and drop:", "        if skip < items:")

    t.r(CACHE, '''        # store and its byte count, shareable between two encoder nodes of one process (C2: `share`)
        self._shared = {"size": 0, "store": OrderedDict()}
''', '''        self.size = 0
        self.store: OrderedDict[tuple, list[torch.Tensor]] = OrderedDict()
''')
    t.cut(CACHE, "    @property\n    def store(self)", "    @staticmethod\n    def key(", new="")
    t.r(CACHE, "        keys = self.keys_for(pixels, grids, extra, digests)\n", '''        if digests is not None:
            keys = [("given", d, g, str(p.dtype), tuple(p.shape[1:]), *extra)
                    for d, p, g in zip(digests, pixels, grids, strict=True)]
        else:
            keys = [self.key(p, g, extra) for p, g in zip(pixels, grids, strict=True)]
''')


def strip_c1(t: Tree) -> None:
    t.f["mstar/model/components/c1_registry.py"] = None
    t.cut(ENG, "from mstar.model.components import c1_registry  # noqa: E402  (mm_reuse_study C1)\n",
          "# the lean set of a step", new="")
    t.r(ENG, "                _c1_check(full, prefix.tokens if prefix is not None else 0, batch.node_name, walk)\n")
    t.r(ENG, "                if c1_registry.ENABLED:   # mm_reuse_study C1\n"
             "                    self._c1_publish(rid, batch.node_name, walk, leases)\n")
    t.cut(ENG, "    def _c1_publish(", "    def prefix_cache_open", new="")
    t.r(ENG, "            _c1_check(inputs, 0, batch.node_name, walk)\n")
    t.r(ENG, "        _c1_check(inputs, matched, batch.node_name, walk)\n")

    t.cut("mstar/engine/resources/kv/manager.py", "    def c1_covered_items(",
          "    @staticmethod\n    def _lease_of", new="")
    cond = "mstar/conductor/conductor.py"
    t.r(cond, "\nfrom mstar.model.components import c1_registry  # noqa: E402  (mm_reuse_study C1)\n")
    t.r(cond, "        request_data.c1_request_id = body.request_id   # mm_reuse_study C1\n")
    t.cut(cond, "        if c1_registry.ENABLED and with_tensors:", "        pstate.cursor = k\n", new="")

    t.r(MODEL, "from mstar.model.components import c1_registry  # mm_reuse_study C1\n")
    t.cut(MODEL, "    if c1_registry.ENABLED:\n        # mm_reuse_study C1: the leading text",
          "    for group in _media_groups", new="")
    t.cut(MODEL, "            c1 = [{}] * len(prefills)\n", "            # the last prefill samples", new="")
    t.r(MODEL, "FINAL_PREFILL: k == len(prefills) - 1, **c1[k],", "FINAL_PREFILL: k == len(prefills) - 1,")

    t.r(SUB, "import os\nimport time\n", "import os\n")
    t.r(SUB, 'from mstar.model.components import c1_registry  # noqa: E402  (mm_reuse_study C1)\n'
             'LOG_C1 = os.environ.get("MSTAR_C1_LOG")\n')
    t.r(SUB, "        self._c1_served: dict = {}   # mm_reuse_study C1\n")
    t.cut(SUB, "        if LOG_C1:\n            with open(LOG_C1, \"a\") as f:\n                f.write(f\"{time.time():.6f}\\t{fwd_info.request_id}\\tencoder_prepare",
          "        listed = list(", new="")
    # （上一处已删到 listed：含 skip 与全部跳过的分支）
    t.cut(SUB, "        if skip:   # mm_reuse_study C1: the leading images", "        if self.output_cache.enabled:\n", new="")
    t.cut(SUB, "    def _c1_skip(", "    def pop_served_outputs", new="")
    t.cut(SUB, "        c1 = self._c1_served.pop(rid, None)\n", "        full = self.output_cache.pop_served", new="")
    t.r(SUB, '''            rows = inputs[name][0].shape[0]
            if c1_registry.ENABLED and grids and not inputs.get("video_second_per_grid"):
                # mm_reuse_study C1: the encoder may have skipped leading images, which the row still holds
                rows = int(grids[0].reshape(-1, 3).prod(dim=1).sum()) // self.config.vision.spatial_merge_size ** 2
            return InputSeqLenInfo(rows + 2 * items + around)''',
        "            return InputSeqLenInfo(inputs[name][0].shape[0] + 2 * items + around)")
    t.cut(SUB, "        pad = 0\n        if graph_walk == \"prefill_vision\" and c1_registry.ENABLED",
          "        if graph_walk == \"prefill_vision\" and inputs[\"image_grid_thw\"]", new="")
    t.r(SUB, '''            row = self._grouped_image_row(fwd_info, inputs, start_pos)   # mm_reuse_study M3
            if pad:   # the skipped images' rows end where image ``pad``'s BOS sits
                row.c1_pad_end = self._c1_rows_before(fwd_info, inputs, pad)
            return row
''', "            return self._grouped_image_row(fwd_info, inputs, start_pos)   # mm_reuse_study M3\n")
    t.r(SUB, "            if pad:\n                row.c1_pad_end = start + vision_len\n")
    t.cut(SUB, "    def _c1_pad(", "    def _grouped_image_row(", new="")


def strip_m4(t: Tree) -> None:
    t.cut(MODEL, "# mm_reuse_study M4 (MSTAR_MEDIA_KEYS=1)", "\n\nclass _PreprocessCache", new="")
    # （上一处已删到 class _PreprocessCache：含 _key_tensor）
    t.cut(MODEL, "\n\nclass _FileMemo:", "_PREPROCESS_CACHE = _PreprocessCache(", new="\n")
    t.r(MODEL, "_FILE_MEMO = _FileMemo(_PREPROCESS_CACHE.cap if MEDIA_KEYS else 0)\n")
    t.r(MODEL, 'input_names=["pixel_values", "image_grid_thw", *(["image_keys"] if MEDIA_KEYS else [])],',
        'input_names=["pixel_values", "image_grid_thw"],')
    t.r(MODEL, '            **({"image_keys": "image_keys"} if MEDIA_KEYS else {}),   # mm_reuse_study M4\n')
    t.cut(MODEL, "    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:", "    def load_video(", new="")
    t.r(MODEL, '''        # M4: a memoised file needs no conversion (its placeholder carries the processed output)
        pil_images = [None if hasattr(img, "_m4") else as_uint8_chw(img, truncate=True).permute(1, 2, 0).numpy()
                      for img in raw_image_inputs]''',
        "        pil_images = [as_uint8_chw(img, truncate=True).permute(1, 2, 0).numpy() for img in raw_image_inputs]")
    t.r(MODEL, '''        for raw, img in zip(raw_image_inputs, pil_images, strict=True):
            img_proc = self._processor.image_processor
            memo = getattr(raw, "_m4", None)
            if memo is not None:
                key, img_out = memo
            else:
                key = (_PREPROCESS_CACHE.key(img, type(img_proc).__name__)
                       if _PREPROCESS_CACHE.enabled or MEDIA_KEYS else None)
                img_out = _PREPROCESS_CACHE.get(key)   # mm_reuse_study M3b
            if img_out is None:
                # inferred, a 1, 3 or 4 px tall image would be read as channels first
                img_out = img_proc(images=[img], return_tensors="pt", input_data_format="channels_last")
                if MEDIA_KEYS:   # M4: bf16 once, before caching
                    img_out = {**img_out, "pixel_values": img_out["pixel_values"].to(torch.bfloat16)}
                if _PREPROCESS_CACHE.enabled:
                    _PREPROCESS_CACHE.put(key, img_out)
            else:
                _hits += 1
            if memo is None and getattr(raw, "_m4_file", None) is not None:
                _FILE_MEMO.put(raw._m4_file, (key, img_out))
            result["pixel_values"].append(img_out["pixel_values"])
            result["image_grid_thw"] += img_out["image_grid_thw"]
            if MEDIA_KEYS:
                result.setdefault("image_keys", []).append(_key_tensor(key[0]))
        if MEDIA_KEYS and pil_images:
            result["image_inputs"] = []   # M4: replaces the decoded images in the stored tensors
''', '''        for img in pil_images:
            img_proc = self._processor.image_processor
            key = _PREPROCESS_CACHE.key(img, type(img_proc).__name__) if _PREPROCESS_CACHE.enabled else None
            img_out = _PREPROCESS_CACHE.get(key)   # mm_reuse_study M3b
            if img_out is None:
                # inferred, a 1, 3 or 4 px tall image would be read as channels first
                img_out = img_proc(images=[img], return_tensors="pt", input_data_format="channels_last")
                if _PREPROCESS_CACHE.enabled:
                    _PREPROCESS_CACHE.put(key, img_out)
            else:
                _hits += 1
            result["pixel_values"].append(img_out["pixel_values"])
            result["image_grid_thw"] += img_out["image_grid_thw"]
''')
    t.r(MODEL, '''                if MEDIA_KEYS:
                    out[span_name("image_keys", first)] = [torch.cat(
                        [out.pop(span_name("image_keys", i))[0] if i != first
                         else out[span_name("image_keys", i)][0] for i in images])]
''')
    t.r(SUB, '''        listed = list(inputs["pixel_values"])
        pixel_values = listed[0] if len(listed) == 1 else torch.cat(listed, dim=0)
''', '''        pixel_values = torch.cat(list(inputs["pixel_values"]), dim=0)
''')
    t.r(SUB, '''            keys = inputs.get("image_keys")   # M4: keyed by the data worker, so no pixel is read here
            digests = None if not keys else [tuple(r) for k in keys for r in k.reshape(-1, 2).tolist()]
            plan = self.output_cache.plan((fwd_info.request_id, fwd_info.rid_handle), per_image, grid, merge,
                                          digests=digests if digests and len(digests) == len(grid) else None)
''', '''            plan = self.output_cache.plan((fwd_info.request_id, fwd_info.rid_handle), per_image, grid, merge)
''')
    t.r(CACHE, '''             extra: tuple = (), digests: list | None = None) -> ItemPlan:
        """``digests``: per item, a key the caller already has (M4: the data worker's), used
        instead of hashing the pixels; it is paired with the grid, dtype and patch shape all the same."""
        _t0 = time.perf_counter()
        if digests is not None:
            keys = [("given", d, g, str(p.dtype), tuple(p.shape[1:]), *extra)
                    for d, p, g in zip(digests, pixels, grids, strict=True)]
        else:
            keys = [self.key(p, g, extra) for p, g in zip(pixels, grids, strict=True)]
        _t1 = time.perf_counter()
''', '''             extra: tuple = ()) -> ItemPlan:
        keys = [self.key(p, g, extra) for p, g in zip(pixels, grids, strict=True)]
''')
    t.r(CACHE, '''                f.write(f"{time.time():.6f}\\t{ids[0]}\\t{len(keys)}\\t{hits}\\t{len(encode)}"
                        f"\\tkey_ms={(_t1 - _t0) * 1000:.1f}\\tplan_ms={(time.perf_counter() - _t0) * 1000:.1f}\\n")''',
        '''                f.write(f"{time.time():.6f}\\t{ids[0]}\\t{len(keys)}\\t{hits}\\t{len(encode)}\\n")''')


def strip_m3b(t: Tree) -> None:
    t.r(MODEL, "import os\nimport time\nfrom collections import OrderedDict\n", "import os\n")
    t.cut(MODEL, "\n\nclass _PreprocessCache:", "\n\n\ndef _media_groups", new="")
    t.r(MODEL, '''        _t_img, _hits = time.perf_counter(), 0   # mm_reuse_study M3b timing
        for img in pil_images:
            img_proc = self._processor.image_processor
            key = _PREPROCESS_CACHE.key(img, type(img_proc).__name__) if _PREPROCESS_CACHE.enabled else None
            img_out = _PREPROCESS_CACHE.get(key)   # mm_reuse_study M3b
            if img_out is None:
                # inferred, a 1, 3 or 4 px tall image would be read as channels first
                img_out = img_proc(images=[img], return_tensors="pt", input_data_format="channels_last")
                if _PREPROCESS_CACHE.enabled:
                    _PREPROCESS_CACHE.put(key, img_out)
            else:
                _hits += 1
            result["pixel_values"].append(img_out["pixel_values"])
            result["image_grid_thw"] += img_out["image_grid_thw"]
''', '''        for img in pil_images:
            img_proc = self._processor.image_processor
            # inferred, a 1, 3 or 4 px tall image would be read as channels first
            img_out = img_proc(images=[img], return_tensors="pt", input_data_format="channels_last")
            result["pixel_values"].append(img_out["pixel_values"])
            result["image_grid_thw"] += img_out["image_grid_thw"]
''')
    t.cut(MODEL, "        if _PP_LOG and pil_images:", "\n        for audio in np_audios", new="")
    dw = "mstar/api_server/data_worker.py"
    t.r(dw, '_PP_LOG = os.environ.get("MSTAR_PREPROCESS_LOG")   # mm_reuse_study profiling\n\n')
    t.r(dw, "        _pp_t0 = time.perf_counter()   # mm_reuse_study profiling (MSTAR_PREPROCESS_LOG)\n")
    t.r(dw, "        _pp_t1 = time.perf_counter()\n")
    t.r(dw, "            _pp_t2 = time.perf_counter()\n")
    t.cut(dw, "            if _PP_LOG:\n                with open(_PP_LOG, \"a\") as f:\n                    f.write(f\"{time.time():.6f}\\tprepare_input",
          "        elif input.text is not None:", new="")
    t.r(dw, "        _pp_t3 = time.perf_counter()\n")
    t.cut(dw, "        if _PP_LOG:\n            with open(_PP_LOG, \"a\") as f:\n                f.write(f\"{time.time():.6f}\\tstore_persisted",
          "        if stream_items:\n            call.held", new="")


def strip_m3(t: Tree) -> None:
    t.r(MODEL, '"""\n\nimport os\nimport logging', '"""\n\nimport logging')
    t.r(MODEL, '''    for group in _media_groups(plan, media):
        end = len(plan) if group[-1] == media[-1] else group[-1] + 1
        steps.append(ThinkerStep(Qwen3OmniModel._MM_WALKS[plan[group[0]].modality][0], plan[start:end]))''',
        '''    for at in media:
        end = len(plan) if at == media[-1] else at + 1
        steps.append(ThinkerStep(Qwen3OmniModel._MM_WALKS[plan[at].modality][0], plan[start:end]))''')
    t.cut(MODEL, "# mm_reuse_study M3: consecutive images", "class ImageSpan(NamedTuple):", new="")
    t.r(MODEL, '''        steps = _thinker_steps(plan)
        for k, step in enumerate(steps):''', '''        for k, step in enumerate(_thinker_steps(plan)):''')
    t.cut(MODEL, '''        out = split_spans(out, self.MEDIA_SPANS)[0]
        for step in steps:''', "    def prompt_tokens(",
          new="        return split_spans(out, self.MEDIA_SPANS)[0]\n\n")
    t.r(SUB, '''            # M3: a grouped row holds several images, each between its own two sentinels
            grids = inputs.get("image_grid_thw") if graph_walk == "prefill_vision" else None
            items = int(grids[0].reshape(-1, 3).shape[0]) if grids else 1
            return InputSeqLenInfo(inputs[name][0].shape[0] + 2 * items + around)''',
        "            return InputSeqLenInfo(inputs[name][0].shape[0] + 2 + around)")
    t.r(SUB, '''        if graph_walk == "prefill_vision" and inputs["image_grid_thw"][0].reshape(-1, 3).shape[0] > 1:
            return self._grouped_image_row(fwd_info, inputs, start_pos)   # mm_reuse_study M3

''')
    t.cut(SUB, "    def _grouped_image_row(", "    def _media_mask(", new="")


def strip_m1(t: Tree) -> None:
    t.f[CACHE] = None
    t.r(ENG, '''    # mm_reuse_study M1: rows whose whole output a submodule served from its own
    # cache (``pop_served_outputs``); they run no forward and complete like cached_rids
    served_outputs: dict[str, NameToTensorList] = field(default_factory=dict)
''')
    t.cut(ENG, '                pop_served = getattr(submodule, "pop_served_outputs", None)\n',
          "                if req_inputs is not None and probe_prefix:", new="")
    t.r(ENG, "            for rid, served in batch.served_outputs.items():   # mm_reuse_study M1\n"
             "                outputs.per_rid_outputs[rid] = served\n")
    t.r(SUB, "from mstar.model.components.encoder_output_cache import EncoderOutputCache  # noqa: E402  (mm_reuse_study M1)\n\n\n")
    t.r(SUB, "        # mm_reuse_study M1: per-image output cache, opt-in (MSTAR_ENCODER_CACHE_MB)\n"
             "        self.output_cache = EncoderOutputCache()\n")
    t.cut(SUB, '        pixel_values = torch.cat(list(inputs["pixel_values"]), dim=0)\n',
          "        indices, weights = vision_interpolation(", new="")
    t.r(SUB, '"pixel_values": pixel_values,', '"pixel_values": torch.cat(list(inputs["pixel_values"]), dim=0),')
    t.cut(SUB, "    def pop_served_outputs(", "    def _capture_block_loop(", new="")
    t.r(SUB, '''        rows = self._split_rows(
            engine_inputs.request_ids, {self.output: [embeds], "deepstack": deepstack}, row_patches,
        )
        if self.output_cache.enabled:
            # mm_reuse_study M1: put cached images back in prompt order, store the new ones
            for rid, out in rows.items():
                if rid in self.output_cache.plans:
                    full = self.output_cache.assemble(rid, [out[self.output][0], *out["deepstack"]])
                    rows[rid] = {self.output: [full[0]], "deepstack": list(full[1:])}
        return rows
''', '''        return self._split_rows(
            engine_inputs.request_ids, {self.output: [embeds], "deepstack": deepstack}, row_patches,
        )
''')
    b = "mstar/model/bagel/submodules.py"
    t.r(b, "from mstar.model.components.encoder_output_cache import EncoderOutputCache  # noqa: E402  (mm_reuse_study M1)\n\n\n")
    t.r(b, "        # mm_reuse_study M1: per-image output cache, opt-in (MSTAR_ENCODER_CACHE_MB)\n"
           "        self.output_cache = EncoderOutputCache()\n")
    t.cut(b, "        if self.output_cache.enabled:   # mm_reuse_study M1: key on the decoded image",
          '        if image_preprocess == "vllm":', new="")
    t.r(b, '''        rid = engine_inputs.request_ids[0]
        if rid in self.output_cache.plans:   # mm_reuse_study M1
            return {"img_emb": self.output_cache.assemble(rid, [features])}
''')
    t.r(b, '''            if rid in self.output_cache.plans:   # mm_reuse_study M1
                out[rid] = {"img_emb": self.output_cache.assemble(rid, out[rid]["img_emb"])}
''')
    t.cut(b, "\n    def pop_served_outputs(", "\n\ndef _unguided(", new="")


def strip_m0(t: Tree) -> None:
    t.r(SUB, "import os\nimport functools", "import functools")
    block = '''        # mm_reuse_study M0: on 5880 (Ada, 100 KB shared memory) the Talker's capture sizing hits a
        # Triton kernel that needs 144 KB; text-only runs never use these nodes, so skip their graphs.
        if os.environ.get("MSTAR_SPEECH_NODES_EAGER") == "1":
            return []
'''
    text = t.f[SUB]
    assert text.count(block) == 2, text.count(block)
    t.f[SUB] = text.replace(block, "")


STRIPS = {"diag": strip_diag, "C4": strip_c4, "C2": strip_c2, "C1": strip_c1, "M4": strip_m4,
          "M3b": strip_m3b, "M3": strip_m3, "M1": strip_m1, "M0": strip_m0}


def git(*args, cwd=REPO, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    a = p.parse_args()
    release = git("describe", "--tags", "--exact-match", "HEAD", check=False).strip()
    if not release:
        release = git("rev-parse", "--short", "HEAD")
    out = Path(a.out) if a.out else ROOT / "patches" / f"mstar-series-{release}"
    files = {f: (REPO / f).read_text() if (REPO / f).exists() else None for f in FILES}
    t = Tree(files)
    snaps = {"final": t.snapshot()}
    for _, code, _ in reversed(STAGES):
        STRIPS[code](t)
        snaps[code] = t.snapshot()   # 剥掉 code 之后（= code 之前的状态）
        with tempfile.TemporaryDirectory() as d:
            for f, text in t.f.items():
                if text is not None and f.endswith(".py"):
                    (Path(d) / "x.py").write_text(text)
                    py_compile.compile(str(Path(d) / "x.py"), doraise=True)
    for f, text in t.f.items():
        head = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=REPO, capture_output=True, text=True)
        base = head.stdout if head.returncode == 0 else None
        if text != base:
            import difflib
            diff = "".join(difflib.unified_diff((base or "").splitlines(True), (text or "").splitlines(True), f, f, n=1))
            raise SystemExit(f"剥完后 {f} 与 HEAD 不同：\n{diff[:3000]}")
    print("剥离完整：全部剥完与 HEAD 相同")

    # 正序：状态 k = 剥掉 STAGES[k:] 之后
    order = [code for _, code, _ in STAGES]
    states = [snaps[order[0]]] + [snaps[order[k + 1]] for k in range(len(order) - 1)] + [snaps["final"]]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wt = Path(tempfile.mkdtemp(prefix="mstar-series-"))
    git("worktree", "add", "--detach", str(wt), "HEAD")
    try:
        git("config", "user.name", "mm_reuse_study", cwd=wt)
        git("config", "user.email", "study@localhost", cwd=wt)
        for k, (num, code, title) in enumerate(STAGES):
            after = states[k + 1]
            if after == states[k]:
                continue
            for f, text in after.items():
                path = wt / f
                if text is None:
                    if path.exists():
                        git("rm", "-q", f, cwd=wt)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                    git("add", f, cwd=wt)
            git("commit", "-q", "-m", f"mm_reuse_study {code}: {title}", cwd=wt)
            patch = git("format-patch", "-1", "--stdout", "HEAD", cwd=wt)
            (out / f"{num}-{code}.patch").write_text(patch)
        full = git("diff", "HEAD~%d" % len(list(out.glob("*.patch"))), "HEAD", cwd=wt)
        (out.parent / f"mstar-{release}-study.patch").write_text(full)
    finally:
        git("worktree", "remove", "--force", str(wt))
    for f in sorted(out.glob("*.patch")):
        print(f.name, sum(1 for line in f.read_text().splitlines() if line.startswith(("+", "-"))
                          and not line.startswith(("+++", "---"))), "行改动")


if __name__ == "__main__":
    sys.exit(main())
