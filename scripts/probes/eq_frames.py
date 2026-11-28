import json, sys, requests
sys.path.insert(0, ".")
from bench.stream import make_frame, frame_part, SYSTEM
B="http://localhost:18512"; out=sys.argv[1]
res=[]
for sid,n in (("eqA",3),("eqB",8),("eqC",8),("eqD",20)):
    for q in ("Describe the colors of these images in order, one word per image.", "How many images are there and what is the color of the last one?"):
        content=[frame_part(make_frame(sid,i,2)[0]) for i in range(n)]+[{"type":"text","text":q}]
        body={"model":"qwen3_omni","messages":[{"role":"system","content":SYSTEM},{"role":"user","content":content}],"max_tokens":48,"modalities":["text"],"temperature":0}
        r=requests.post(B+"/v1/chat/completions",json=body).json()
        res.append({"sid":sid,"n":n,"q":q[:20],"text":r["choices"][0]["message"]["content"]})
json.dump(res, open(out,"w"), indent=1); print("\n".join(f"{x['sid']} {x['n']} {x['text'][:90]!r}" for x in res))
