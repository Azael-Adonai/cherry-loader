import base64
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = json.load(open(os.path.join(D, "meta.json")))
for it in meta["items"]:
    raw = open(os.path.join(D, "sprites", it["file"]), "rb").read()
    it["data"] = "data:image/webp;base64," + base64.b64encode(raw).decode()

html = open(os.path.join(D, "index.html"), encoding="utf-8").read()
inject = "<script>window.__CHERRY_META = " + json.dumps(meta) + ";</script>\n<script>"
html = html.replace("<script>", inject, 1)

os.makedirs(os.path.join(D, "dist"), exist_ok=True)
out = os.path.join(D, "dist", "cherry_loader_single.html")
open(out, "w", encoding="utf-8").write(html)
print("cherry_loader_single.html:", os.path.getsize(out) // 1024, "KB")
