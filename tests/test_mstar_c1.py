"""M* 原型 C1（前缀已覆盖的图不编码）的 CPU 测试：Thinker 补零的行数与位置、补零区域的结束行、登记文件的读写。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "mstar"))
from mstar.model.components import c1_registry  # noqa: E402
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule  # noqa: E402

MERGE = 2


def fake_thinker(before_len: int):
    cfg = SimpleNamespace(vision=SimpleNamespace(spatial_merge_size=MERGE))
    t = SimpleNamespace(config=cfg, _text_around=lambda fwd, inp: (torch.zeros(before_len), torch.zeros(0)))
    return t


def test_pad_puts_zeros_for_skipped_leading_images():
    grids = torch.tensor([[1, 4, 4], [1, 4, 8], [1, 4, 4]])   # 4, 8, 4 行
    kept = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 1   # 只剩第 3 张
    inputs = {"image_grid_thw": [grids], "vision_embeds": [kept], "deepstack": [kept * 2, kept * 3]}
    out, skipped = ThinkerSubmodule._c1_pad(fake_thinker(0), inputs, "cpu")
    assert skipped == 2
    assert out["vision_embeds"][0].shape == (16, 3)
    assert torch.count_nonzero(out["vision_embeds"][0][:12]) == 0
    assert torch.equal(out["vision_embeds"][0][12:], kept)
    assert all(torch.equal(d[12:], kept * k) for d, k in zip(out["deepstack"], (2, 3)))


def test_pad_noop_when_nothing_skipped():
    grids = torch.tensor([[1, 4, 4]])
    e = torch.ones(4, 3)
    inputs = {"image_grid_thw": [grids], "vision_embeds": [e], "deepstack": [e]}
    out, skipped = ThinkerSubmodule._c1_pad(fake_thinker(0), inputs, "cpu")
    assert skipped == 0 and out is inputs


def test_pad_end_row_of_grouped_row():
    # 行：前置文本 5 个；图 i 占 BOS + n_i + EOS。跳过前 2 张（4、8 行）后，补零区到第 2 张的最后一个视觉行为止
    grids = torch.tensor([[1, 4, 4], [1, 4, 8], [1, 4, 4]])
    inputs = {"image_grid_thw": [grids]}
    end = ThinkerSubmodule._c1_rows_before(fake_thinker(5), None, inputs, 2)
    # 图 0：BOS 在 5，视觉行 6..9，EOS 10；图 1：BOS 11，视觉行 12..19，EOS 20 -> 补零区结束（不含）= 20
    assert end == 20


def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(c1_registry, "DIR", str(tmp_path / "c1"))
    assert c1_registry.read("req#1") == 0
    c1_registry.write("req#1", 32)
    assert c1_registry.read("req#1") == 32
    c1_registry.drop("req#1")
    assert c1_registry.read("req#1") == 0
