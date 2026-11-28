# M\* C4 轻量版：摄像头会话的帧引用与预热提示（5880，2026-11-28）

设计：DESIGN 18.4 / 18.7 的 C4（用户选的轻量版：不改请求模型，仍是一问一个请求）。开关 `MSTAR_C4=1`，叠加在 C2 段（M1 + M3(16) + M3b + M4 + C1 + C2，`configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml`）之上。补丁 `patches/mstar-series/07-C4.patch`。客户端 `bench/stream.py --c4`。

## 做法

1. 服务端第一次加载某帧时（`Qwen3OmniModel.load_image`），把它的字节存到 `MSTAR_C4_FRAMES_DIR/<xxh3-128>`（默认 /dev/shm/mstar_c4_frames）。
2. 客户端对已成功送达的帧只发引用：一个很小的 data URL，内容为 `MSTAR-FRAME:<xxh3-128>`。模型据此找回原帧（M4 文件记忆命中时连文件也不读）；data worker 把引用按原帧的字节算前缀键（`_c4_original`），所以整帧发送与引用发送共用同一批 KV 页。
3. 请求顶层字段 `c4_warm = n`（M\* 聊天端点把未知顶层字段当作 model_kwargs）：前 n 张图整段在本会话较早的请求里。模型把这 n 张图并成一个图像步（不受 16 张一组的限制），C1 整组跳过，像素不发（`MSTAR_C4_KEEP` 张例外，默认 0）。其余图照常每 16 张一组。
4. 提示不成立（租约没覆盖这 n 张图）时，视觉编码器报错，客户端去掉提示、整帧重发一次。
5. 客户端的 n = 与上一个成功请求的公共帧前缀 − 1：那个请求的最后一帧可能止于半页，不整段在缓存里。

## 正确性

`eq-c4.json`（`scripts/probes/eq_c4.py`）：8 个提问分别整帧发送、引用 + 提示发送，16 次都与参照 `../q3-m3/eq-g3pp.json` 逐字相同；`eq.json`：不开 C4 的客户端在开了 C4 的服务上也逐字相同。

## 调试中发现的三个问题（各自的服务日志）

| 目录 | 问题 | 修正 |
|---|---|---|
| — | 提示放在 `model_kwargs` 里被忽略（日志 "Ignoring model_kwargs this model does not read"） | 作为顶层字段发送 |
| `server-hintbug/`、`server-dbg/`、`smoke2.jsonl`、`dbg.jsonl` | 提示按最近一个请求算时几乎每个请求都重发（`dbg.jsonl` 10 个预热中 8 个），租约总比提示少一帧（`server-dbg/c1.log` 的 publish 行） | 当时改成按最近两个请求中较早的算、并保留最后 4 张图的像素（`MSTAR_C4_KEEP=4`）——见下节，这只是在补下一行的错误 |
| `server-keybug/` | 完全重复的请求报 "images 0+ were sent without pixels, but the prefix lease holds only 0"：引用文件按自身字节算了前缀键，与整帧发送时的键不同 | data worker 按原帧算键（`_c4_original`） |

第二行的根因其实是第三行：一帧在第 k 个请求里第一次整帧发送，在第 k+1 个请求里作为引用发送，键不同，所以前缀正好少最后那一帧。

## 提示规则对照（`hint-keep0/`，`scripts/probes/c4_hint.sh`）

同一服务，`MSTAR_C4_KEEP=0`（预热组全部不带像素），W 臂 timing，`--c4-hint latest` 与 `older` 交错，grow、slide 各 2 个会话；先跑一个丢弃的会话预热。

| 提问 TTFT ms（2 个会话） | grow q0/q1/q2 | slide q0/q1/q2/q3 |
|---|---|---|
| latest（按最近一个请求） | 140,123 / 119,110 / 122,114 | 115,130 / 171,114 / 119,131 / 135,125 |
| older（按较早的请求） | 137,132 / 114,104 / 137,140 | 128,128 / 102,117 / 120,120 / 128,123 |

283 个请求（含预热）0 次重发，16 个提问全部正确。两种规则的 TTFT 在噪声内相同（预热组只差一帧）。结论：修正引用键之后，按最近一个请求提示、预热组全部不带像素就成立；`older` 与 `KEEP=4` 不再需要。现在的默认是 `latest`、`MSTAR_C4_KEEP=0`。G3 的 C4 轮（`results/g3-stream/5880/mstar/stream-c4/`）用的是旧规则，脚本段 `mstar7` 固定旧规则以便复现。

## 单会话冒烟（早期，供参考）

`smoke.jsonl`（`server-hintbug/` 上第一个会话）、`smoke3.jsonl`（`server-keybug/`，引用键修正之前；提示按较早请求，所以没有重发）：W 臂 16/32/48 帧提问 TTFT 159/206/246 与 140/149/170 ms，请求体 2.6/4.9/7.1 KB。正式数字以 G3 C4 轮为准（112–130 ms）。

## 结论

- W 臂提问的 TTFT 不再随帧数增长：已预热的帧一个 walk（C1 跳过）+ 新帧与问题一个 walk。G3 中 48 帧 232 → 120 ms，7 个提问点 6 个快于 vLLM-Omni。
- 改动：模型 107 行、编码器 12 行、data worker 17 行（补丁 07），客户端约 40 行；没改引擎与调度器。
- 局限：帧存放没有淘汰（/dev/shm 随会话增长，服务重启时清空）；提示由客户端给，服务端只校验不推断；完整版 C4（一个请求内流式追加帧）没做。
