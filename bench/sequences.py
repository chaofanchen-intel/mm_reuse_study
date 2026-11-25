"""image→text 的请求序列 S1–S4（DESIGN 5.1）。

每个序列是一个生成器：yield 一个 Req，runner 发出请求后把回答 send 回来（S4 多轮对话要用上一轮的回答）。

图像在代码里确定性生成（不需要夹具文件）：主色 + 低幅度噪声（按图像 id 播种），所以
- 每张图的字节不同（内容哈希不同，不会被别的序列或别的臂命中），
- 主色可以作为确定答案，用来核对回答（DESIGN 10.3）。
图像 id 包含 臂/序列/重复/分辨率，同一服务上跑多个序列和臂不会互相污染。
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
from PIL import Image

PALETTE = {
    "red": (200, 40, 40), "green": (40, 160, 60), "blue": (40, 60, 200), "yellow": (230, 210, 40),
    "purple": (130, 50, 160), "orange": (240, 140, 30), "black": (20, 20, 20), "white": (240, 240, 240),
}
SYNONYMS = {"purple": ("purple", "violet", "magenta"), "orange": ("orange",), "white": ("white",),
            "black": ("black",)}

Q_COLOR = "What is the main color of this image? Answer with one word."
QUESTIONS = [
    Q_COLOR,
    "Is this image mostly a single color? Answer yes or no.",
    "Describe this image in five words.",
    "Is there any text in this image? Answer yes or no.",
    "Give this image a short title.",
]
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a careful assistant that answers briefly.",
    "You are an image analysis assistant. Be concise.",
    "You are a helpful assistant. Answer in English.",
    "You are a visual assistant for a design team.",
]
SP_DEFAULT = SYSTEM_PROMPTS[0]
S4_FOLLOWUPS = [
    ("Is that a warm or a cool color? Answer with one word.", None),
    ("Name one object that often has this color.", None),
    ("Answer again: what is the main color of the image? One word.", "COLOR"),
]

_cache: dict[tuple[str, int], tuple[bytes, str]] = {}


def make_image(image_id: str, res: int) -> tuple[bytes, str]:
    """返回 (PNG 字节, 主色名)。同一 (id, res) 结果相同。"""
    key = (image_id, res)
    if key not in _cache:
        h = int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8], "little")
        color = list(PALETTE)[h % len(PALETTE)]
        rng = np.random.default_rng(h)
        base = np.array(PALETTE[color], dtype=np.int16)
        noise = rng.integers(-6, 7, size=(res, res, 3), dtype=np.int16)
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        _cache[key] = (buf.getvalue(), color)
    return _cache[key]


def color_correct(answer: str, color: str) -> bool:
    a = answer.lower()
    return any(s in a for s in SYNONYMS.get(color, (color,)))


@dataclass
class Req:
    seq: str
    repeat: int
    step: int                      # 序列内的序号
    res: int
    system_prompt: str
    question: str
    image_ids: list[str]           # 本请求里出现的图像
    new_image: bool                # 图像是否第一次出现在本服务上（由序列构造保证）
    expect: str | None             # 期望答案（主色）；None 表示不核对
    messages: list[dict] = field(default_factory=list)   # OpenAI 消息（图像字段保留为占位，runner 填入）


def _user(image_id: str | None, text: str) -> dict:
    content = []
    if image_id is not None:
        content.append({"type": "image", "image_id": image_id})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def s1_same_image_questions(tag: str, repeat: int, res: int, n: int = 5) -> Generator[Req, str, None]:
    img = f"{tag}|s1|r{repeat}|{res}|A"
    color = make_image(img, res)[1]
    for i in range(n):
        q = QUESTIONS[i % len(QUESTIONS)]
        msgs = [{"role": "system", "content": SP_DEFAULT}, _user(img, q)]
        yield Req("s1", repeat, i, res, SP_DEFAULT, q, [img], i == 0, color if q == Q_COLOR else None, msgs)


def s2_same_image_system_prompts(tag: str, repeat: int, res: int, n: int = 5) -> Generator[Req, str, None]:
    img = f"{tag}|s2|r{repeat}|{res}|A"
    color = make_image(img, res)[1]
    for i in range(n):
        sp = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
        msgs = [{"role": "system", "content": sp}, _user(img, Q_COLOR)]
        yield Req("s2", repeat, i, res, sp, Q_COLOR, [img], i == 0, color, msgs)


def s3_new_images(tag: str, repeat: int, res: int, n: int = 5) -> Generator[Req, str, None]:
    for i in range(n):
        img = f"{tag}|s3|r{repeat}|{res}|N{i}"
        color = make_image(img, res)[1]
        msgs = [{"role": "system", "content": SP_DEFAULT}, _user(img, Q_COLOR)]
        yield Req("s3", repeat, i, res, SP_DEFAULT, Q_COLOR, [img], True, color, msgs)


def s4_multi_turn(tag: str, repeat: int, res: int) -> Generator[Req, str, None]:
    img = f"{tag}|s4|r{repeat}|{res}|A"
    color = make_image(img, res)[1]
    msgs = [{"role": "system", "content": SP_DEFAULT}, _user(img, Q_COLOR)]
    answer = yield Req("s4", repeat, 0, res, SP_DEFAULT, Q_COLOR, [img], True, color, list(msgs))
    for i, (q, exp) in enumerate(S4_FOLLOWUPS, start=1):
        msgs = msgs + [{"role": "assistant", "content": answer or ""}, _user(None, q)]
        answer = yield Req("s4", repeat, i, res, SP_DEFAULT, q, [img], False,
                           color if exp == "COLOR" else None, list(msgs))


SEQUENCES = {"s1": s1_same_image_questions, "s2": s2_same_image_system_prompts,
             "s3": s3_new_images, "s4": s4_multi_turn}


def warmup_requests(tag: str, res: int, n: int = 3) -> list[Req]:
    out = []
    for i in range(n):
        img = f"{tag}|warmup|{res}|W{i}"
        color = make_image(img, res)[1]
        msgs = [{"role": "system", "content": SP_DEFAULT}, _user(img, Q_COLOR)]
        out.append(Req("warmup", 0, i, res, SP_DEFAULT, Q_COLOR, [img], True, color, msgs))
    return out
