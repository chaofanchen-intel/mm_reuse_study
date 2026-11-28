"""C4 路径的逐字对照：每个 eq_frames 的提示先完整发一次（让服务端存帧、写前缀），再用帧引用 + c4_warm 发一次，
两次回答都与参照（q3-m3/eq-g3pp.json）比较。服务需 MSTAR_C4=1。用法：python scripts/probes/eq_c4.py OUT.json"""
import json, sys, time
import requests
sys.path.insert(0, ".")
from bench.stream import make_frame, frame_part, c4_ref_part, SYSTEM

B = "http://localhost:18512"
out = sys.argv[1]
ref = json.load(open("results/smoke/5880/mstar/q3-m3/eq-g3pp.json"))
res, k = [], 0
for sid, n in (("eqA", 3), ("eqB", 8), ("eqC", 8), ("eqD", 20)):
    for q in ("Describe the colors of these images in order, one word per image.",
              "How many images are there and what is the color of the last one?"):
        frames = [make_frame(sid, i, 2)[0] for i in range(n)]
        for mode in ("full", "c4"):
            content = [(c4_ref_part(f) if mode == "c4" else frame_part(f)) for f in frames] + [{"type": "text", "text": q}]
            body = {"model": "qwen3_omni", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                    "max_tokens": 48, "modalities": ["text"], "temperature": 0}
            if mode == "c4":
                body["c4_warm"] = max(0, n - 2)
            r = requests.post(B + "/v1/chat/completions", json=body).json()
            text = r["choices"][0]["message"]["content"] if "choices" in r else f"ERROR {r}"
            res.append({"sid": sid, "n": n, "q": q[:20], "mode": mode, "text": text, "same_as_ref": text == ref[k]["text"]})
            time.sleep(0.5)
        k += 1
json.dump(res, open(out, "w"), indent=1)
print("\n".join(f"{x['sid']} {x['n']:2d} {x['mode']:4s} {'same' if x['same_as_ref'] else 'DIFF'} {x['text'][:60]!r}" for x in res))
