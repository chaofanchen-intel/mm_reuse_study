| 框架 | 臂 | 序列 | 分辨率 | 状态 | n | TTFT 中位 ms | TTFT IQR | 编码器 ms 中位 | 前缀命中中位 | 输出 token 中位 | 正确率 | 警告 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mstar | enc0-pfx0 | s1 | 512 | cov=none enc=run | 40 | 285.6 | 270.1–290.0 | 20.5 | - | 11.0 | 100% (8) | 0 |
| mstar | enc0-pfx0 | s1 | 980 | cov=none enc=run | 40 | 852.5 | 839.0–861.5 | 55.7 | - | 9.0 | 100% (8) | 0 |
| mstar | enc0-pfx0 | s2 | 512 | cov=none enc=run | 40 | 284.5 | 269.7–292.6 | 19.1 | - | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx0 | s2 | 980 | cov=none enc=run | 40 | 853.4 | 843.4–860.2 | 56.0 | - | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx0 | s3 | 512 | cov=none enc=run | 40 | 283.8 | 271.5–289.2 | 19.4 | - | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx0 | s3 | 980 | cov=none enc=run | 40 | 856.0 | 843.0–862.3 | 56.0 | - | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx0 | s4 | 512 | cov=none enc=run | 32 | 285.2 | 270.0–292.9 | 19.9 | - | 2.0 | 100% (16) | 0 |
| mstar | enc0-pfx0 | s4 | 980 | cov=none enc=run | 32 | 857.1 | 844.4–864.8 | 55.5 | - | 2.0 | 100% (16) | 0 |
| mstar | enc0-pfx1 | s1 | 512 | cov=full enc=run | 12 | 102.9 | 91.2–105.1 | 23.7 | 1385.0 | 11.0 | - | 0 |
| mstar | enc0-pfx1 | s1 | 512 | cov=none enc=run | 8 | 284.9 | 278.5–291.0 | 19.2 | 0.0 | 2.0 | 100% (8) | 0 |
| mstar | enc0-pfx1 | s1 | 512 | cov=partial enc=run | 20 | 123.6 | 112.0–136.5 | 23.7 | 1280.0 | 11.0 | - | 0 |
| mstar | enc0-pfx1 | s1 | 980 | cov=full enc=run | 12 | 154.5 | 152.2–161.3 | 54.4 | 4916.0 | 11.0 | - | 0 |
| mstar | enc0-pfx1 | s1 | 980 | cov=none enc=run | 8 | 844.3 | 839.1–849.9 | 55.5 | 0.0 | 2.0 | 100% (8) | 0 |
| mstar | enc0-pfx1 | s1 | 980 | cov=partial enc=run | 20 | 195.1 | 186.8–201.0 | 59.2 | 4864.0 | 11.0 | - | 0 |
| mstar | enc0-pfx1 | s2 | 512 | cov=none enc=run | 40 | 279.9 | 268.8–290.1 | 20.8 | 0.0 | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx1 | s2 | 980 | cov=none enc=run | 40 | 849.8 | 841.1–856.5 | 55.7 | 0.0 | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx1 | s3 | 512 | cov=none enc=run | 40 | 279.3 | 269.8–290.4 | 19.2 | 0.0 | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx1 | s3 | 980 | cov=none enc=run | 40 | 848.3 | 835.7–861.2 | 55.8 | 0.0 | 2.0 | 100% (40) | 0 |
| mstar | enc0-pfx1 | s4 | 512 | cov=full enc=run | 19 | 98.2 | 86.7–104.4 | 20.9 | 1408.0 | 2.0 | 100% (8) | 0 |
| mstar | enc0-pfx1 | s4 | 512 | cov=none enc=run | 8 | 286.2 | 268.4–292.1 | 19.7 | 0.0 | 2.0 | 100% (8) | 0 |
| mstar | enc0-pfx1 | s4 | 512 | cov=partial enc=run | 5 | 128.5 | 127.3–134.8 | 26.7 | 1280.0 | 2.0 | - | 0 |
| mstar | enc0-pfx1 | s4 | 980 | cov=full enc=run | 9 | 161.9 | 159.4–173.1 | 58.5 | 4916.0 | 2.0 | 100% (3) | 0 |
| mstar | enc0-pfx1 | s4 | 980 | cov=none enc=run | 8 | 859.6 | 852.6–863.6 | 55.9 | 0.0 | 2.0 | 100% (8) | 0 |
| mstar | enc0-pfx1 | s4 | 980 | cov=partial enc=run | 15 | 198.8 | 195.7–201.6 | 57.3 | 4864.0 | 2.0 | 100% (5) | 0 |
| vllm-omni | enc1-pfx0 | s1 | 512 | cov=none enc=cache | 32 | 216.5 | 207.0–219.7 | - | - | 6.5 | - | 0 |
| vllm-omni | enc1-pfx0 | s1 | 512 | cov=none enc=run | 8 | 254.3 | 238.7–260.3 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s1 | 980 | cov=none enc=cache | 32 | 889.3 | 885.0–892.2 | - | - | 5.5 | - | 0 |
| vllm-omni | enc1-pfx0 | s1 | 980 | cov=none enc=run | 8 | 1059.2 | 1056.2–1060.3 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s2 | 512 | cov=none enc=cache | 32 | 217.2 | 209.9–220.1 | - | - | 2.0 | 100% (32) | 0 |
| vllm-omni | enc1-pfx0 | s2 | 512 | cov=none enc=run | 8 | 256.7 | 247.5–260.2 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s2 | 980 | cov=none enc=cache | 32 | 890.2 | 886.6–896.6 | - | - | 2.0 | 100% (32) | 0 |
| vllm-omni | enc1-pfx0 | s2 | 980 | cov=none enc=run | 8 | 1049.4 | 1047.3–1059.7 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s3 | 512 | cov=none enc=run | 40 | 254.4 | 239.1–257.8 | - | - | 2.0 | 100% (40) | 0 |
| vllm-omni | enc1-pfx0 | s3 | 980 | cov=none enc=run | 40 | 1054.4 | 1048.2–1061.2 | - | - | 2.0 | 100% (40) | 0 |
| vllm-omni | enc1-pfx0 | s4 | 512 | cov=none enc=cache | 24 | 230.6 | 221.4–231.8 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s4 | 512 | cov=none enc=run | 8 | 258.6 | 255.8–262.4 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s4 | 980 | cov=none enc=cache | 24 | 895.0 | 883.2–900.1 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx0 | s4 | 980 | cov=none enc=run | 8 | 1061.4 | 1056.5–1066.4 | - | - | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s1 | 512 | cov=none enc=run | 8 | 254.6 | 239.5–260.3 | - | 0.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s1 | 512 | cov=partial enc=cache | 32 | 67.1 | 64.6–70.8 | - | 1344.0 | 6.5 | - | 0 |
| vllm-omni | enc1-pfx1 | s1 | 980 | cov=none enc=run | 8 | 1050.0 | 1043.1–1059.4 | - | 0.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s1 | 980 | cov=partial enc=cache | 32 | 100.6 | 94.8–101.9 | - | 4864.0 | 6.5 | - | 0 |
| vllm-omni | enc1-pfx1 | s2 | 512 | cov=none enc=cache | 32 | 210.3 | 197.8–218.5 | - | 0.0 | 2.0 | 100% (32) | 0 |
| vllm-omni | enc1-pfx1 | s2 | 512 | cov=none enc=run | 8 | 248.9 | 235.0–262.4 | - | 0.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s2 | 980 | cov=none enc=cache | 32 | 887.7 | 882.5–893.4 | - | 0.0 | 2.0 | 100% (32) | 0 |
| vllm-omni | enc1-pfx1 | s2 | 980 | cov=none enc=run | 8 | 1065.2 | 1057.5–1069.9 | - | 0.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s3 | 512 | cov=none enc=run | 40 | 245.8 | 238.9–258.3 | - | 0.0 | 2.0 | 100% (40) | 0 |
| vllm-omni | enc1-pfx1 | s3 | 980 | cov=none enc=run | 40 | 1058.9 | 1043.6–1065.4 | - | 0.0 | 2.0 | 100% (40) | 0 |
| vllm-omni | enc1-pfx1 | s4 | 512 | cov=full enc=skip | 16 | 69.7 | 62.9–73.2 | - | 1408.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s4 | 512 | cov=none enc=run | 8 | 258.5 | 256.5–260.8 | - | 0.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s4 | 512 | cov=partial enc=cache | 8 | 67.7 | 64.3–70.8 | - | 1344.0 | 2.0 | - | 0 |
| vllm-omni | enc1-pfx1 | s4 | 980 | cov=full enc=skip | 24 | 97.5 | 94.7–102.3 | - | 4928.0 | 2.0 | 100% (8) | 0 |
| vllm-omni | enc1-pfx1 | s4 | 980 | cov=none enc=run | 8 | 1064.8 | 1058.9–1074.9 | - | 0.0 | 2.0 | 100% (8) | 0 |
