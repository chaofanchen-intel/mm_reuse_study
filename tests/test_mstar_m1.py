"""M* 原型 M1（编码器输出缓存）的 CPU 测试：命中/未命中拆分、按提示顺序重组、至少编码一项、LRU 上限。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "mstar"))
from mstar.model.components.encoder_output_cache import EncoderOutputCache  # noqa: E402

MERGE = 2


def fake_encode(pixels: list[torch.Tensor], grids):
    """每张图的输出 = 其像素和（可区分），行数 = t*h*w/merge²；两个 deepstack 为其倍数。"""
    outs = []
    for p, (t, h, w) in zip(pixels, grids):
        n = t * h * w // MERGE ** 2
        outs.append(torch.full((n, 4), float(p.sum())))
    e = torch.cat(outs)
    return [e, e * 2, e * 3]


def run(cache, rid, pixels, grids):
    plan = cache.plan((rid, 7), pixels, grids, MERGE)
    assert plan.encode, "test helper expects at least one miss"
    fresh = fake_encode([pixels[i] for i in plan.encode], [grids[i] for i in plan.encode])
    return plan, cache.assemble(rid, fresh)


def test_hits_and_order():
    c = EncoderOutputCache(cap_mb=1)
    grids = [(1, 4, 4), (1, 4, 8), (1, 4, 4)]
    px = [torch.randn(t * h * w, 3) for t, h, w in grids]
    ref = fake_encode(px, grids)
    p1, out1 = run(c, "a", px[:2], grids[:2])
    assert p1.encode == [0, 1]
    p2, out2 = run(c, "b", px, grids)          # 前两张命中，第三张新
    assert p2.encode == [2]
    assert all(torch.equal(x, y) for x, y in zip(out2, ref))
    p3 = c.plan(("c", 9), px, grids, MERGE)    # 全部命中：不编码，按任一 id 取走，两个 id 都清掉
    assert p3.encode == [] and not c.plans
    out3 = c.pop_served(9)
    assert all(torch.equal(x, y) for x, y in zip(out3, ref))
    assert c.pop_served("c") is None and not c.served


def test_lru_cap():
    c = EncoderOutputCache(cap_mb=0.001)       # ~1 KB：每项 4x4 float32 x3 = 192 B
    grids = [(1, 4, 4)] * 8
    px = [torch.randn(16, 3) for _ in grids]
    run(c, "a", px, grids)
    assert c.size <= c.cap and len(c.store) < 8


def test_m4_given_digests_skip_pixels():
    """M4：给定键时不读像素——像素不同、键相同仍命中；键不同则不命中。"""
    cache = EncoderOutputCache(64)
    grids = [(1, 4, 4), (1, 4, 4)]
    px = [torch.rand(16, 8), torch.rand(16, 8)]
    plan = cache.plan(("r1", 1), px, grids, MERGE, digests=[(1, 2), (3, 4)])
    assert plan.encode == [0, 1]
    cache.assemble("r1", fake_encode(px, grids))
    other = [torch.zeros(16, 8), torch.zeros(16, 8)]        # 像素不同，键相同
    plan = cache.plan(("r2", 2), other, grids, MERGE, digests=[(1, 2), (3, 4)])
    assert plan.encode == [] and cache.pop_served("r2") is not None
    plan = cache.plan(("r3", 3), px, grids, MERGE, digests=[(1, 2), (9, 9)])
    assert plan.encode == [1]


def test_c2_shared_store_between_two_caches():
    a, b = EncoderOutputCache(cap_mb=1), EncoderOutputCache(cap_mb=1)
    b.share(a)
    grids = [(1, 4, 4)]
    px = [torch.randn(16, 3)]
    run(a, "x", px, grids)                       # a 编码并存入
    plan = b.plan(("y", 1), px, grids, MERGE)    # b 直接命中
    assert not plan.encode and b.pop_served("y") is not None
    assert a.size == b.size > 0
    assert b.keys_for(px, grids)[0] in a.store
