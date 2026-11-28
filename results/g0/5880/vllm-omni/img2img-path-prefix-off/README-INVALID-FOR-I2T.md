# 这个目录的聊天用例不是图像理解

这一轮请求没有带 `"modalities": ["text"]`。vLLM-Omni 的 BAGEL 在不带时走图像路径（img2img + DiT 阶段）：
日志里的 key 是 `pixel_values_img2img`，每个请求约 22 s，回答是乱码。因此：

- 这里的 ViT 次数、前缀命中（2368/4818）、回答质量，**都不是 image→text 的结果**，不能引用。
- 其中的 t2i 与 edit 行仍然是对应任务的结果（edit：同一参考图第二次 ViT 0 次）。
- image→text 的正确结果在 `results/g0/5880/vllm-omni/i2t/`，说明见 `results/g0/5880/vllm-omni/NOTES.md`。
