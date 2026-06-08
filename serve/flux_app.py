"""FLUX.1-schnell low-latency image-gen server (stdlib HTTP, no FastAPI).

Loads a FLUX (or SDXL-Turbo) diffusers pipeline ONCE, serves a single-file web UI
at GET /, and a JSON image endpoint at POST /api/generate that returns a base64 PNG.
Optional SFW LoRA loading via load_lora_weights. One generation at a time (GPU lock).

Launch:
  HF_HOME=/workspace/hf_cache /workspace/flux_venv/bin/python serve/flux_app.py \
      --model black-forest-labs/FLUX.1-schnell --port 8888
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_STATE: dict = {"pipe": None, "model": None, "kind": "flux", "loras": []}
_GEN_LOCK = threading.Lock()
_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux.html")

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def load_pipeline(model: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    kind = "flux" if "flux" in model.lower() else "sdxl"
    print(f"[flux] loading {model} (kind={kind}) ...", flush=True)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    )
    pipe = pipe.to("cuda")
    # low-latency / memory: VAE slicing + tiling (cheap, no quality loss)
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _STATE.update(pipe=pipe, model=model, kind=kind)
    print(f"[flux] ready: {model}", flush=True)
    return pipe


def load_lora(repo: str, weight_name: str | None, adapter: str):
    pipe = _STATE["pipe"]
    kwargs = {"adapter_name": adapter}
    if weight_name:
        kwargs["weight_name"] = weight_name
    pipe.load_lora_weights(repo, token=os.environ.get("HF_TOKEN"), **kwargs)
    if adapter not in _STATE["loras"]:
        _STATE["loras"].append(adapter)
    print(f"[flux] loaded LoRA {repo} as '{adapter}'", flush=True)


def generate(req: dict) -> dict:
    import torch

    pipe = _STATE["pipe"]
    kind = _STATE["kind"]
    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("empty prompt")
    steps = int(req.get("steps", 4 if kind == "flux" else 4))
    steps = max(1, min(steps, 50))
    # schnell + turbo are guidance-distilled -> guidance 0; allow override
    guidance = float(req.get("guidance", 0.0))
    width = int(req.get("width", 1024))
    height = int(req.get("height", 1024))
    width = max(256, min(width, 1536)) // 8 * 8
    height = max(256, min(height, 1536)) // 8 * 8
    seed = req.get("seed")
    gen = None
    if seed is not None and str(seed) != "":
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

    kw = dict(prompt=prompt, num_inference_steps=steps,
              guidance_scale=guidance, width=width, height=height)
    if gen is not None:
        kw["generator"] = gen
    neg = (req.get("negative_prompt") or "").strip()
    if neg and kind != "flux":   # flux schnell ignores negative prompt
        kw["negative_prompt"] = neg

    t0 = time.time()
    image = pipe(**kw).images[0]
    dt = time.time() - t0

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image": b64, "seconds": round(dt, 2), "steps": steps,
            "width": width, "height": height, "model": _STATE["model"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[http] " + (fmt % args), flush=True)

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/api/health":
            self._send(200, json.dumps({
                "ok": _STATE["pipe"] is not None,
                "model": _STATE["model"], "kind": _STATE["kind"],
                "loras": _STATE["loras"],
            }).encode())
            return
        if self.path == "/" or self.path.startswith("/?"):
            try:
                with open(_HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"flux.html not found")
            return
        self._send(404, json.dumps({"error": "not found"}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}).encode())
            return
        path = self.path.rstrip("/")
        if path == "/api/lora":
            try:
                load_lora(req["repo"], req.get("weight_name"),
                          req.get("adapter", "lora"))
                self._send(200, json.dumps({"ok": True, "loras": _STATE["loras"]}).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode())
            return
        if path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}).encode())
            return
        if not _GEN_LOCK.acquire(blocking=False):
            self._send(429, json.dumps({"error": "busy: a generation is running"}).encode())
            return
        try:
            out = generate(req)
            self._send(200, json.dumps(out).encode())
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode())
        finally:
            _GEN_LOCK.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--lora", default=None, help="optional HF LoRA repo to preload")
    ap.add_argument("--lora-weight", default=None)
    a = ap.parse_args()
    load_pipeline(a.model)
    if a.lora:
        load_lora(a.lora, a.lora_weight, "lora")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[flux] listening on http://0.0.0.0:{a.port}  (GET / , POST /api/generate)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[flux] shutting down", flush=True)


if __name__ == "__main__":
    main()
