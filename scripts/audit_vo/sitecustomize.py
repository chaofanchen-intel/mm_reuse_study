"""vLLM-Omni 测量 shim（不改框架代码）。由 scripts/serve_vllm_omni_cuda.sh --shim count|time 开启。

环境变量：
- VO_AUDIT_LOG：日志路径。没有这个变量时什么也不做。
- VO_AUDIT_MODE：count（默认）只记录调用；time 额外在编码器前后做设备同步并记录耗时（会影响 TTFT，只用于画像轮）。

记录（制表符分隔：时间、pid、类型、详情）：
- encode_vit       BAGEL 的 OmniBagelForConditionalGeneration._encode_vit（每次 ViT 前向一行）
- embed_multimodal BAGEL 的 embed_multimodal（每次批量编码一行，含多模态 key，用于核对任务路径）
- ecm_check        vLLM EncoderCacheManager.check_and_update_cache 的结果（hit、mm_hash 前缀、请求 id 前缀）
- mm_encoder       vLLM ModelState.execute_mm_encoder（与模型无关；每个调度步一行：编码项数、请求 id 前缀；
                   time 模式加 ms）。TP>1 时每个 rank 一行，按 pid 区分。
- prefix_lookup    vLLM KVCacheManager.get_computed_blocks（调度器接纳请求时）：请求 id、prompt token 数、前缀命中 token 数。
                   聚合计数器 vllm:prefix_cache_* 在请求紧接请求时会延迟到下一个请求才发布，按请求归因要用这一行。
                   同一请求等待期间可能被查询多次，按请求 id 取最后一次。
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
_MS = "vllm.v1.worker.gpu.model_states.interface"
_KVM = "vllm.v1.core.kv_cache_manager"


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


def _patch_model_state(module):
    cls = getattr(module, "ModelState", None)
    if cls is None:
        return
    orig = cls.execute_mm_encoder

    def execute_mm_encoder(self, scheduled_encoder_inputs):
        items = sum(len(v) for v in scheduled_encoder_inputs.values())
        reqs = ",".join(sorted(r[:20] for r in scheduled_encoder_inputs))
        if not items:
            return orig(self, scheduled_encoder_inputs)
        if not _TIME:
            _record("mm_encoder", f"items={items} reqs={reqs}")
            return orig(self, scheduled_encoder_inputs)
        _sync()
        t0 = time.perf_counter()
        out = orig(self, scheduled_encoder_inputs)
        _sync()
        _record("mm_encoder", f"items={items} reqs={reqs} ms={(time.perf_counter() - t0) * 1000:.3f}")
        return out

    cls.execute_mm_encoder = execute_mm_encoder


def _patch_kvm(module):
    cls = getattr(module, "KVCacheManager", None)
    if cls is None:
        return
    orig = cls.get_computed_blocks

    def get_computed_blocks(self, request, *args, **kwargs):
        out = orig(self, request, *args, **kwargs)
        _record("prefix_lookup", f"req={request.request_id[:40]} prompt={request.num_prompt_tokens} hit={out[1]}")
        return out

    cls.get_computed_blocks = get_computed_blocks


_PATCHERS = {_BAGEL: _patch_bagel, _ECM: _patch_ecm, _MS: _patch_model_state, _KVM: _patch_kvm}


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
