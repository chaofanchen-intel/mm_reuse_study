# 本研究对框架源码的改动

- 全部改动：`git -C external/mstar apply ../../patches/mstar-v0.3.0-study.patch`；`git -C external/vllm-omni apply ../../patches/vllm-omni-v0.34.0-study.patch`。
- 按改动看：`patches/mstar-series-v0.3.0/` 是按顺序叠加的系列补丁（`00-M0` … `08-diag`），每个只含一项改动，在上游 `v0.3.0` 上依次 `git apply`（或 `git am`）。依赖：M4 需 M1、M3；C1 需 M4；C2 需 M1、C1；C4 需 C1 与 M4 的文件记忆（`MSTAR_PREPROCESS_CACHE_MB` > 0）。
- 由 `scripts/patches/split_mstar_series.py` 从工作树生成：倒序逐项剥离，每步检查能编译，剥完必须与上游逐字节相同；依次打上系列补丁与工作树逐字节相同。改了 M\* 源码后重新运行它，系列补丁和 `mstar-<tag>-study.patch` 总补丁会一并生成。
- 每项都有开关（下表），默认关时行为与上游相同，所以"某项的效果"可以在同一份全量代码上只切开关来测。
服务启动清单记录 `framework_diff_sha256`（`git diff HEAD` 的哈希）与未跟踪文件数，结果可以对回补丁版本。

| 补丁 | 编号 | 内容 | 开关（默认关，关时行为与上游相同） |
|---|---|---|---|
| `mstar-v0.3.0-study.patch` | M0 | Qwen3-Omni 的 Talker/Code2Wav 不建 CUDA graph（5880 共享内存 100 KB，Talker 的 Triton 核要 144 KB，启动失败；文本输出不用这两个节点） | `MSTAR_SPEECH_NODES_EAGER=1` |
| 同上 | M1 | 编码器输出缓存：Qwen3-Omni 视觉编码器按图像缓存输出（键：像素 xxh3-128 + 网格 + dtype；存放在锁页主存，命中时拷回 GPU——M* 的显存规划不知道这块缓存，放在 GPU 上 2 GB 时 rank 0 OOM）；一行的图像全部命中时，引擎把该行当作"缓存服务"，不跑前向，直接路由缓存的输出（`engine.py` 中标 "mm_reuse_study M1" 的三处） | `MSTAR_ENCODER_CACHE_MB` > 0；`MSTAR_ENCODER_CACHE_LOG` 记录每行的图像数/命中/编码数（缓存关时也记录） |
| 同上 | M1（BAGEL） | 同一缓存用于 BAGEL 的 ViT（键另含预处理参数）；缓存的主机侧方法加 `torch.compiler.disable`（BAGEL 的 ViT 在 torch.compile 内调用它们，Dynamo 不支持 `pin_memory`） | 同上 |
| 同上 | M3 | Qwen3-Omni 相邻图像（中间无文本）最多 N 张合成一个 walk：一次视觉编码、Thinker 一行内依次排布每张图的哨兵、位置与 deepstack（与逐图 walk 的位置和 KV 相同）；每张图仍有自己的前缀键 | `MSTAR_GROUP_IMAGES=N`。默认 Thinker 每步预算 1024 token 时 N ≤ 3：更大的组超过预算被分块 prefill，分块 + 前缀缓存 + TP2 出错（第二次请求 0 命中，随后 "TP ranks diverged"）。用 `configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml`（预算 16384）时 N = 16 已验证；N = 48 时视觉编码一次 5.8 万 patch，GPU 0 OOM |
| 同上 | M3b | 图像预处理缓存：HF image processor 的输出按解码后像素的 xxh3-128 缓存（主存 LRU） | `MSTAR_PREPROCESS_CACHE_MB`；`MSTAR_PREPROCESS_LOG` 记录每个 prompt 的图像数、命中数与耗时，以及 data worker 每请求的解码 / prompt 处理 / 键 / 存张量耗时与字节数 |
| 同上 | M4 | 已缓存媒体不再搬数据：data worker 给每张图算一次键（解码像素的 xxh3-128）随图发出，视觉编码器的 M1 查找不再读、不再哈希像素；像素以 bf16 发送（patch embed 本来就转 bf16，结果逐字相同）；不再存放 Qwen3-Omni 无节点读取的解码图像（每帧 3.7 MB fp32）；同一文件字节（xxh3-128）第二次出现时跳过 JPEG 解码与处理 | `MSTAR_MEDIA_KEYS=1`（需要 `MSTAR_PREPROCESS_CACHE_MB` > 0 才有文件记忆） |
| 同上 | C1 | 前缀租约已覆盖的图不编码、不传：开头文本单独一步，Thinker rank 0 取租约后登记覆盖的图数（/dev/shm/mstar_c1，原型通道），调度器对整组覆盖的步不发像素，编码器跳过、Thinker 补零，引擎检查补零区在命中范围内 | `MSTAR_C1=1`（日志 `MSTAR_C1_LOG`，serve 脚本自动设） |
| 同上 | C2 | 编码提前：`VisionAhead` 分区 + 第二个 ViT 节点 `vision_ahead`（与主编码器同进程，共享 M1 缓存），请求进入即编码各组，与 Thinker prefill 重叠 | `MSTAR_C2=1`，需 M1 与配置 `qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml` |
| 同上 | C4 | 摄像头会话（轻量版，不改请求模型）：服务端首次加载的帧字节存到 `MSTAR_C4_FRAMES_DIR`（默认 /dev/shm/mstar_c4_frames），之后客户端只发引用（data URL 内容为 `MSTAR-FRAME:<帧字节 xxh3-128>`，约 60 字节），data worker 把引用按原帧算前缀键；请求顶层字段 `c4_warm = n` 表示前 n 张图整段在本会话较早的请求里，它们合成一组由 C1 跳过，其中除最后 `MSTAR_C4_KEEP` 张外不带像素；租约实际不覆盖时请求报错，客户端整帧重发 | `MSTAR_C4=1`（`MSTAR_C4_KEEP` 默认 0）；客户端 `bench/stream.py --c4` |
| 同上 | 诊断 | Thinker prefill 步的同步计时（待完成工作 / 发射 / GPU 尾部、是否回放图）；C1 登记的每次租约覆盖数 | `MSTAR_FWD_LOG=<路径>`；`MSTAR_C1_LOG` |
| `vllm-omni-v0.34.0-study.patch` | V1 | WS 视频流端点在音频输出会话中也做 warmup：放开 `_incremental_prefill_active` 的音频限制；warmup 的引擎输出只要文本（只跑 thinker 阶段），提示仍按会话的 modalities 渲染，token 与提问一致 | `VLLM_OMNI_AUDIO_WARMUP=1` |

测试：`tests/test_mstar_m1.py`（CPU，缓存逻辑）；GPU 验证见 `results/smoke/5880/mstar/q3-stream/NOTES.md`。
