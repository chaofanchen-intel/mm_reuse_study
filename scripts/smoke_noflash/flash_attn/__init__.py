# smoke 用：屏蔽 flash_attn，让 BAGEL ViT 走 SDPA 回退（经 PYTHONPATH 作用于所有子进程）
raise ImportError("flash_attn blocked for the no-flash smoke arm")
