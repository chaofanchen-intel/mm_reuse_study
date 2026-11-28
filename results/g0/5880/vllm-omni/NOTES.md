# G0 audit: vLLM-Omni 0.34.0 BAGEL (2026-11-25，修正版)

## 配置
`vllm serve <BAGEL-7B-MoT snapshot> --omni --port 18401 --deploy-config configs/bagel_vllm_omni_audit.yaml`
= 上游 bagel.yaml，改动三处（都由 47 GB 显存卡强制）：阶段 0 `enable_prefix_caching: true`（上游默认关）；
阶段 1（DiT）放 GPU 3（上游与阶段 0 同卡，同卡启动 OOM）；阶段 0 `gpu_memory_utilization` 0.85（上游 0.48，在本卡上 KV 为 0）。
ViT 次数与耗时来自 `scripts/audit_vo/sitecustomize.py`（环境变量开启的包装，不改框架代码；耗时带 cuda 同步）。

## 重要：任务必须显式指定
image→text 请求必须带 `"modalities": ["text"]`，否则 BAGEL 走 img2img + DiT（约 22 s，输出乱码）。
上一轮没带，结果作废，见 `results/g0/5880/vllm-omni/img2img-path-prefix-on/README-INVALID-FOR-I2T.md`。

## image→text（正确任务）`results/g0/5880/vllm-omni/i2t/requests.jsonl`，512x512 红色图，temperature 0
| label | E2E ms | 前缀命中 / 查询 token | ViT 次数 | 回答 |
|---|---|---|---|---|
| A-image-first-q0（冷） | 837 | 0 / 1383 | 1 | Red |
| A-image-first-q1..q3（同图，不同问题） | 167, 86, 159 | 1344 / ~1383 | 0 | 正确 |
| B-text-first-q0, q1（同图，文本在前） | 57, 153 | 1344 / ~1383 | 0 | 正确 |
| A-new-image-B（新图） | 247 | 0 / 1383 | 1 | Blue |
| t2i-512 | 19022 | 0 / 8 | 0 | — |

- 编码器缓存：同图重复请求时命中（ecm_check hit=True，稳定 mm_hash），ViT 0 次。新图 miss，ViT 一次约 27 ms（518x518 输入）。
- 前缀缓存：命中 1344 个 token（块粒度，查询 1383），新图 0。
- 文本在前（B）与图像在前（A）得到同样的命中量，疑似 prompt 构造把图像放在前面；未验证（见 DESIGN 第 11 节）。
- 前三个 ViT 记录是启动期（980x980，约 107–130 ms），不是请求。

## 编辑（img2img）`results/g0/5880/vllm-omni/img2img-path-prefix-on/requests.jsonl` 的 i2i 行
同一参考图第二次编辑：ViT 0 次（编码器缓存命中）；第一次 1 次。单次编辑约 22 s（含 DiT）。
