"""Realtime training dashboard — stdlib HTTP server (no extra deps).

Serves a live browser page that polls the metrics file train.py writes and renders
loss / LR / throughput charts updating in realtime.

  python src/monitor/server.py --metrics checkpoints/eurollm_cluster/metrics.jsonl --port 8888

On RunPod: pod exposes 8888/http -> open the proxy URL. Locally: http://localhost:8888
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

METRICS_PATH = "checkpoints/eurollm_cluster/metrics.jsonl"

INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Project Berlin — Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{background:#0b0e14;color:#d7dce5;font:14px/1.4 system-ui,sans-serif;margin:0;padding:20px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#6b7280;margin-bottom:16px}
 .stats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
 .card{background:#141925;border:1px solid #232a3a;border-radius:10px;padding:12px 16px;min-width:120px}
 .card .k{color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
 .card .v{font-size:22px;font-weight:600;margin-top:2px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .chart{background:#141925;border:1px solid #232a3a;border-radius:10px;padding:12px;height:280px}
 @media(max-width:880px){.grid{grid-template-columns:1fr}}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:6px}
 .stale{background:#ef4444}
</style></head><body>
<h1>🇩🇪 Project Berlin — EuroLLM Diffusion Training</h1>
<div class=sub><span id=live class=dot></span><span id=status>connecting…</span></div>
<div class=stats id=stats></div>
<div class=grid>
 <div class=chart><canvas id=lossC></canvas></div>
 <div class=chart><canvas id=lrC></canvas></div>
 <div class=chart><canvas id=tokC></canvas></div>
 <div class=chart><canvas id=gpuC></canvas></div>
</div>
<script>
const mk=(id,label,color)=>new Chart(document.getElementById(id),{type:'line',
 data:{labels:[],datasets:[{label,data:[],borderColor:color,backgroundColor:color+'22',
 borderWidth:2,pointRadius:0,tension:.25,fill:true}]},
 options:{animation:false,responsive:true,maintainAspectRatio:false,
 scales:{x:{ticks:{color:'#6b7280'},grid:{color:'#1c2230'}},
 y:{ticks:{color:'#6b7280'},grid:{color:'#1c2230'}}},
 plugins:{legend:{labels:{color:'#d7dce5'}}}}});
const loss=mk('lossC','loss','#f59e0b'),lr=mk('lrC','learning rate','#60a5fa'),
 tok=mk('tokC','tokens/sec','#22c55e'),gpu=mk('gpuC','GPU mem (GB)','#a78bfa');
let lastT=0;
function stat(k,v){return `<div class=card><div class=k>${k}</div><div class=v>${v}</div></div>`}
async function tick(){
 try{
  const r=await fetch('/metrics.json?_='+Date.now());const m=await r.json();
  const st=document.getElementById('status'),dot=document.getElementById('live');
  if(!m.length){st.textContent='waiting for first step…';return}
  const last=m[m.length-1];
  const labels=m.map(d=>d.step);
  loss.data.labels=labels;loss.data.datasets[0].data=m.map(d=>d.loss);loss.update();
  lr.data.labels=labels;lr.data.datasets[0].data=m.map(d=>d.lr);lr.update();
  tok.data.labels=labels;tok.data.datasets[0].data=m.map(d=>d.tok_per_sec);tok.update();
  gpu.data.labels=labels;gpu.data.datasets[0].data=m.map(d=>d.gpu_gb);gpu.update();
  const fresh=(Date.now()/1000-last.t)<30;dot.className='dot'+(fresh?'':' stale');
  st.textContent=fresh?'training live':'stale (no update >30s)';
  const pct=last.max_steps?((last.step/last.max_steps*100).toFixed(1)+'%'):'';
  document.getElementById('stats').innerHTML=
   stat('step',last.step+(last.max_steps?'/'+last.max_steps:''))+
   stat('progress',pct)+stat('loss',last.loss)+
   stat('lr',last.lr.toExponential(2))+
   stat('tok/s',(last.tok_per_sec/1e3).toFixed(1)+'k')+
   stat('GPU GB',last.gpu_gb)+stat('epoch',last.epoch);
 }catch(e){document.getElementById('status').textContent='server error: '+e}
}
setInterval(tick,2000);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    metrics_path = METRICS_PATH

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path.startswith("/metrics.json"):
            rows = []
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path) as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                rows.append(json.loads(ln))
                            except json.JSONDecodeError:
                                pass
            self._send(200, json.dumps(rows), "application/json")
        else:
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")

    def log_message(self, *a):
        pass  # quiet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=METRICS_PATH)
    ap.add_argument("--port", type=int, default=8888)
    args = ap.parse_args()
    Handler.metrics_path = args.metrics
    print(f"[monitor] serving http://0.0.0.0:{args.port}  (metrics: {args.metrics})")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
