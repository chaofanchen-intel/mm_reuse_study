"""vLLM-Omni 测量 shim（不改框架代码）。由 scripts/serve_vllm_omni_cuda.sh --shim count|time 开启。

环境变量：
- VO_AUDIT_LOG：日志路径。没有这个变量时什么也不做。
- VO_AUDIT_MODE：count（默认）只记录调用；time 额外在编码器前后做设备同步并记录耗时（会影响 TTFT，只用于画像轮）。

记录（制表符分隔：时间、pid、类型、详情）：
- encode_vit       BAGEL 的 OmniBagelForConditionalGeneration._encode_vit（每次 ViT 前向一行）
- embed_multimodal BAGEL 的 embed_multimodal（每次批量编码一行，含多模态 key，用于核对任务路径）
- ecm_check        vLLM EncoderCacheManager.check_and_update_cache 的结果（hit、mm_hash 前缀、请求 id 前缀）
"""
import importlib.abc
import importlib.util
import os
import sys
import time

_LOG = os.environ.get("VO_AUDIT_LOG")
_TIME = os.environ.get("VO_AUDIT_MODE", "count") == "time"
_BAGEL = "vllm_omni.model_executor.models.bagel.bagel"
_ECM = "vllm.v1.core.encoder_cache_manager"


def _record(kind, detail):
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.6f}\t{os.getpid()}\t{kind}\t{detail}\n")


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _patch_bagel(module):
    cls = getattr(module, "OmniBagelForConditionalGeneration", None)
    if cls is None:
        return
    orig_vit, orig_emb = cls._encode_vit, cls.embed_multimodal

    def _encode_vit(self, pixel_values):
        shape = tuple(pixel_values.shape)
        if not _TIME:
            _record("encode_vit", f"shape={shape}")
            return orig_vit(self, pixel_values)
        _sync()
        t0 = time.perf_counter()
        out = orig_vit(self, pixel_values)
        _sync()
        _record("encode_vit", f"shape={shape} ms={(time.perf_counter() - t0) * 1000:.3f}")
        return out

    def embed_multimodal(self, **kwargs):
        keys = sorted(k for k, v in kwargs.items() if v is not None)
        _record("embed_multimodal", f"keys={','.join(keys)}")
        return orig_emb(self, **kwargs)

    cls._encode_vit = _encode_vit
    cls.embed_multimodal = embed_multimodal


def _patch_ecm(module):
    cls = getattr(module, "EncoderCacheManager", None)
    if cls is None:
        return
    orig = cls.check_and_update_cache

    def check_and_update_cache(self, request, input_id):
        hit = orig(self, request, input_id)
        ident = request.mm_features[input_id].identifier
        _record("ecm_check", f"hit={hit} input_id={input_id} req={request.request_id[:24]} mm_hash={ident[:24]}")
        return hit

    cls.check_and_update_cache = check_and_update_cache


_PATCHERS = {_BAGEL: _patch_bagel, _ECM: _patch_ecm}


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHERS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _PATCHERS[name](module)

        spec.loader.exec_module = exec_module
        return spec


if _LOG:
    sys.meta_path.insert(0, _Hook())
