# 场景

每个场景对应 `bench/plan.py` 中的一条 `Scenario`。输入夹具（图片、视频、音频、prompt 列表）放在本目录下，按场景名分子目录，夹具文件应可由脚本再生成，不手工编辑。

待建立：
- `phase0-profile-bagel/`：阶段 0 的图片与 prompt。
- `h1-same-image-n-prompts/`：同一参考图 + N 个 prompt。
- `h4-video-fixed-text-varies/`：固定视频 + 文本变体。
