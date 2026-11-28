import os
import time, requests, sys, json
from bench.stream import make_frame, frame_part, SYSTEM
B="http://localhost:18512"
def req(n, sid, mt=1, q="Color?"):
    content=[frame_part(make_frame(sid,i,6,640,480)[0]) for i in range(n)]+([{"type":"text","text":q}] if q else [])
    body={"model":"qwen3_omni","messages":[{"role":"system","content":SYSTEM},{"role":"user","content":content}],"max_tokens":mt,"modalities":["text"],"temperature":0}
    t=time.time(); r=requests.post(B+"/v1/chat/completions",json=body); assert r.ok, r.text[:200]; return (time.time()-t)*1000, len(json.dumps(body))
if os.environ.get("PROF_WARM"):   # 先用别的帧走一遍冷路径（首次 eager 大 prefill 有约 1.6 s 的一次性开销）
    print("warm", req(48, "warmup-%d" % time.time(), q=None)); time.sleep(1)
for n in [int(x) for x in sys.argv[1].split(",")]:
    sid="m4p%d"%n
    print(n, "cold", req(n,sid,q=None)); time.sleep(1)
    for k in range(3):
        print(n, "cached", req(n,sid,q=None)); time.sleep(1)
# 增长：n 帧之后 n+1 帧（同一会话），看前缀是否在一个分组 walk 内被切开
if len(sys.argv) > 2:
    for n in [int(x) for x in sys.argv[2].split(",")]:
        sid = "grow%d" % n
        print(n, "grow base", req(n, sid, q=None)); time.sleep(1)
        for k in range(1, 4):
            print(n + k, "grow +1", req(n + k, sid, q=None)); time.sleep(1)
