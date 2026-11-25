# 旧版 G0 审计脚本（只作记录，不再使用）

`results/g0/5880/` 下的早期结果由这两个脚本产生：

- `g0_audit_mstar.py` → `mstar/i2t-unsettled`（未等待计数器，归因错误）、`mstar/i2t-settled-repeats`（每个请求都是更早一轮的完全重复）、`mstar/edits`
- `g0_audit_vllm_omni.py` → `vllm-omni/img2img-path-prefix-on`、`vllm-omni/img2img-path-prefix-off`（聊天用例没带 `modalities`，不是 image→text）、`vllm-omni/i2t`

它们各写一套请求与解析，输出格式不同，也不是流式的。新的测量用 `bench/runner.py`（image→text 的 S1–S4）与 `bench/audit/g0_bagel_gen.py`（t2i、编辑），共用 `bench/client.py` 与 `bench/adapters/`。
运行旧脚本需要在本目录下执行（`g0_audit_vllm_omni.py` 从同目录导入 `g0_audit_mstar`）。
