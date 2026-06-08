#!/usr/bin/env python3
"""diffusion-chat backend — streams the reverse-diffusion *denoising reveal* over SSE.

A tiny stdlib http.server (ThreadingHTTPServer, no FastAPI/uvicorn) that:

  * GET  /             -> serves the sibling chat.html frontend
  * GET  /api/health   -> {"ok": true, "mode": ..., "model": ...}
  * POST /api/chat     -> Server-Sent Events; one event PER DENOISING STEP, each
                          carrying the FULL current generation with still-masked
                          positions rendered as the literal char U+2592 ("▒"),
                          then a final {"type":"done", ...} event.

Two model backends, selected by --mode / MODE env, loaded ONCE at startup:

  MODE=berlin  Our Project-Berlin checkpoint. Loads base Qwen/Qwen2.5-0.5B,
               resize_token_embeddings (mean_resizing=False), loads the trained
               safetensors shard(s) from --ckpt (stripping the "_orig_mod." compile
               prefix), and applies apply_blockwise_bidirectional. --ckpt may be a
               concrete stepN dir OR 'latest' (resolves the newest
               checkpoints/qwen_long/stepN at startup). Sampling reuses the EXACT
               confidence-ordered parallel-unmasking loop from
               src/diffusion/generate.py, yielding the partial sequence at EACH
               denoising step. When the request block_size >= gen_len the whole gen
               region is ONE block: full bidirectional attention + a global,
               confidence-ordered, out-of-order reveal across the full sequence.

  MODE=llada   GSAI-ML/LLaDA-8B-Instruct (trust_remote_code, bf16, cuda). Implements
               LLaDA's low-confidence semi-autoregressive remasking generation
               (mask_id=126336, semi-AR blocks, steps_per_block = steps // num_blocks,
               each step unmask the top get_num_transfer_tokens highest-confidence
               positions; temperature via Gumbel noise on the logits). Yields the
               partial decode each step.

CLI:
  python app.py --mode berlin --ckpt latest --port 8888
  python app.py --mode berlin --ckpt checkpoints/qwen_long/step7400 \
      --base Qwen/Qwen2.5-0.5B --port 8890
  python app.py --mode llada --port 8890

One request at a time (global generation lock); model loaded once at startup.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- make src/diffusion importable so we reuse the EXACT project modules -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DIFFUSION_DIR = os.path.join(_REPO, "src", "diffusion")
if _DIFFUSION_DIR not in sys.path:
    sys.path.insert(0, _DIFFUSION_DIR)

# Heavy deps (torch / transformers) are imported lazily inside load so that
# --help and import-time checks work on a machine without them installed.

# The literal char the frontend animates ▒ -> word. U+2592 MEDIUM SHADE.
MASK_CHAR = "▒"
LLADA_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_ID = 126336
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B"

# Cap SSE step events so the UI animates smoothly even for steps >> the cap.
# berlin's single-block global denoise wants MORE granular frames so the reveal
# looks gradual (top-confidence positions trickle in), so it gets a higher cap.
MAX_STEP_EVENTS = 80
BERLIN_MAX_STEP_EVENTS = 120

# Where trained berlin checkpoints live (stepN dirs written by src/diffusion/train.py).
BERLIN_CKPT_ROOT = "checkpoints/qwen_long"

# One generation at a time. The model is shared, single-GPU; serialize requests.
_GEN_LOCK = threading.Lock()

# Populated by load_backend() at startup.
_STATE: dict = {"mode": None, "model_name": None, "backend": None}


# =============================================================================
# Rendering helpers
# =============================================================================
def render_partial(tok, gen_ids, mask_id: int) -> tuple[str, int]:
    """Decode the current generation tensor (1D list/tensor of token ids) to a
    string, replacing every still-masked position with MASK_CHAR. Contiguous
    revealed runs are decoded together so multi-byte tokens join correctly.

    Returns (text, revealed_count). The text is the FULL current generation
    (prompt NOT included)."""
    ids = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)
    out_parts: list[str] = []
    run: list[int] = []
    revealed = 0
    for t_id in ids:
        if t_id == mask_id:
            if run:
                out_parts.append(tok.decode(run, skip_special_tokens=True))
                run = []
            out_parts.append(MASK_CHAR)
        else:
            revealed += 1
            run.append(t_id)
    if run:
        out_parts.append(tok.decode(run, skip_special_tokens=True))
    return "".join(out_parts), revealed


def _emit_steps_filter(total_steps: int, max_events: int = MAX_STEP_EVENTS):
    """Return a predicate emit(step_index_0based, is_last) -> bool that throttles
    to <= max_events events (always emitting the final step)."""
    if total_steps <= max_events:
        return lambda i, is_last: True
    every = math.ceil(total_steps / max_events)
    return lambda i, is_last: is_last or ((i + 1) % every == 0)


# =============================================================================
# Checkpoint resolution
# =============================================================================
def _step_num(path: str) -> int:
    """Extract the integer N from a '.../stepN' checkpoint dir name (-1 if none)."""
    name = os.path.basename(path.rstrip("/"))
    if name.startswith("step") and name[4:].isdigit():
        return int(name[4:])
    return -1


def resolve_latest_ckpt(ckpt: str, root: str = BERLIN_CKPT_ROOT) -> str:
    """Resolve a checkpoint spec to a concrete directory.

    If ckpt is the sentinel 'latest' (case-insensitive), pick the newest
    checkpoints/<root>/stepN dir by the highest N. We honour the LATEST pointer
    file (written by src/diffusion/train.py) when it points at a real dir, then
    fall back to the max-stepN scan. Anything else is returned unchanged so an
    explicit --ckpt path keeps working exactly as before.

    `root` may be absolute or relative; relative is resolved against the repo
    root so the server can be launched from anywhere."""
    if not ckpt or ckpt.strip().lower() != "latest":
        return ckpt

    root_dir = root if os.path.isabs(root) else os.path.join(_REPO, root)
    if not os.path.isdir(root_dir):
        raise RuntimeError(
            f"BERLIN_CKPT=latest but checkpoint root not found: {root_dir!r}")

    # 1) honour the LATEST pointer file if it resolves to a real dir.
    pointer = os.path.join(root_dir, "LATEST")
    if os.path.isfile(pointer):
        try:
            with open(pointer, "r", encoding="utf-8") as f:
                target = f.read().strip()
        except OSError:
            target = ""
        if target:
            cand = target if os.path.isabs(target) else os.path.join(_REPO, target)
            # only trust the pointer for a concrete stepN dir we can load.
            if os.path.isdir(cand) and _step_num(cand) >= 0 \
                    and glob.glob(os.path.join(cand, "*.safetensors")):
                print(f"[berlin] BERLIN_CKPT=latest -> {cand} (via LATEST pointer)",
                      flush=True)
                return cand

    # 2) scan for the highest stepN dir that actually contains weights.
    steps = [d for d in glob.glob(os.path.join(root_dir, "step*"))
             if os.path.isdir(d) and _step_num(d) >= 0
             and glob.glob(os.path.join(d, "*.safetensors"))]
    if not steps:
        raise RuntimeError(
            f"BERLIN_CKPT=latest but no stepN checkpoint with weights under {root_dir!r}")
    newest = max(steps, key=_step_num)
    print(f"[berlin] BERLIN_CKPT=latest -> {newest} "
          f"(newest of {len(steps)} stepN dirs)", flush=True)
    return newest


# =============================================================================
# Backend: BERLIN (our Qwen checkpoint, project generate loop)
# =============================================================================
class BerlinBackend:
    def __init__(self, base: str, ckpt: str, block_size: int = 64):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from attention_surgery import apply_blockwise_bidirectional
        from diffusion import MASK_TOKEN

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        if not ckpt or not os.path.isdir(ckpt):
            raise RuntimeError(f"berlin mode needs a valid --ckpt dir, got: {ckpt!r}")

        # tokenizer (with [MASK]) is saved in the ckpt; arch from base + resize.
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16 if device == "cuda" else None)
        model.resize_token_embeddings(len(tok), mean_resizing=False)

        shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
        if not shards:
            raise RuntimeError(f"no *.safetensors found in {ckpt}")
        state: dict = {}
        for s in shards:
            state.update(load_file(s))
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[berlin] loaded {len(state)} tensors from {len(shards)} shard(s); "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if len(missing) > len(state) * 0.5:
            print("[berlin] WARN: >50% keys missing — near-base model, not trained!",
                  flush=True)

        model.to(device).eval()
        # Remember the configured surgery block size. When a request asks for a
        # single global block (block_size >= gen_len) we re-apply the surgery with
        # a block big enough to make the WHOLE gen region attend bidirectionally
        # (one block); a later non-single-block request restores this default.
        self._default_block_size = block_size
        self._surgery_block_size = block_size
        apply_blockwise_bidirectional(model, block_size=block_size)

        self.model = model
        self.tok = tok
        self.mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)

        # health model field, e.g. "berlin:Qwen2.5-0.5B@step8200".
        base_short = base.split("/")[-1]
        ckpt_tag = os.path.basename(ckpt.rstrip("/"))
        self.model_name = f"berlin:{base_short}@{ckpt_tag}"

    def health_model(self) -> str:
        return self.model_name

    def _ensure_block_size(self, block_size: int) -> None:
        """Re-apply the attention surgery iff the desired block size changed, so a
        single-block global request (block_size >= gen_len) gets FULL bidirectional
        attention across the entire gen region instead of being chopped into the
        default-64 sub-blocks that force a left-to-right-ish reveal."""
        if block_size == self._surgery_block_size:
            return
        from attention_surgery import apply_blockwise_bidirectional
        apply_blockwise_bidirectional(self.model, block_size=block_size)
        self._surgery_block_size = block_size

    @property
    def torch_no_grad(self):
        return self.torch.no_grad

    def generate_stream(self, prompt: str, gen_len: int, steps: int,
                        block_size: int, temperature: float):
        """Yield (kind, payload_dict) tuples. kind in {"step", "done"}.

        Reuses the EXACT cosine-schedule confidence-ordered parallel-unmask loop
        from src/diffusion/generate.py, but yields the partial sequence each step
        instead of only returning the final text.

        SINGLE-BLOCK GLOBAL DENOISE: when the request's block_size >= gen_len the
        whole gen region is treated as ONE block. We (a) re-apply the attention
        surgery so EVERY gen position attends to EVERY other gen position (full
        bidirectional, not the default-64 sub-blocks), and (b) reveal the top
        ceil(remaining_masked / remaining_steps) highest-confidence MASKED
        positions globally each step — out of order across the full sequence, NOT
        left-to-right. Same algorithm as generate.py, just a single block."""
        torch = self.torch
        import torch.nn.functional as F  # noqa: N812

        model, tok, mask_id, device = self.model, self.tok, self.mask_id, self.device

        # single global block iff the caller asked for a block >= the gen region.
        single_block = block_size >= gen_len
        if single_block:
            # one block must cover prompt + gen so the gen region is fully
            # bidirectional (block(k) <= block(q) holds for all k,q in [0,total)).
            self._ensure_block_size(max(gen_len + 256, block_size))
        else:
            # restore the configured default surgery (in case a previous request
            # widened it to a single global block).
            self._ensure_block_size(self._default_block_size)

        with torch.no_grad():
            prompt_ids = (tok(prompt, return_tensors="pt")["input_ids"][0].to(device)
                          if prompt else torch.empty(0, dtype=torch.long, device=device))
            p = prompt_ids.shape[0]
            total = p + gen_len

            seq = torch.full((1, total), mask_id, device=device, dtype=torch.long)
            if p:
                seq[0, :p] = prompt_ids
            gen_slice = slice(p, total)

            # berlin gets a higher event cap so the global reveal looks gradual.
            emit = _emit_steps_filter(steps, BERLIN_MAX_STEP_EVENTS)

            for step in range(steps):
                is_mask = seq[0, gen_slice] == mask_id
                if not is_mask.any():
                    break
                logits = model(input_ids=seq).logits[0, gen_slice]  # (gen_len, vocab)

                # repetition penalty (matches project generate default 1.2)
                rep = 1.2
                if rep != 1.0:
                    present = seq[0, gen_slice][~is_mask]
                    if present.numel():
                        uniq = torch.unique(present)
                        logits[:, uniq] = logits[:, uniq] / rep

                if temperature and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    pred = torch.multinomial(probs, 1).squeeze(-1)
                    conf = probs.gather(-1, pred[:, None]).squeeze(-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)

                remaining_masked = int(is_mask.sum().item())
                conf_masked = conf.clone()
                if single_block:
                    # whole-sequence reveal: top ceil(remaining/remaining_steps)
                    # highest-confidence MASKED positions, globally, out of order.
                    # Rank ONLY masked positions (revealed pushed to the bottom)
                    # so order[:n_reveal] are genuinely the n_reveal highest-conf
                    # masked slots across the FULL sequence (not left-to-right).
                    remaining_steps = steps - step
                    n_reveal = max(1, math.ceil(remaining_masked / remaining_steps))
                    n_reveal = min(n_reveal, remaining_masked)
                    conf_masked[~is_mask] = float("-inf")
                else:
                    # cosine reveal schedule: target #still-masked after this step
                    frac = torch.cos(
                        torch.tensor((step + 1) / steps * torch.pi / 2)).item()
                    keep_masked = int(gen_len * frac)
                    n_reveal = max(1, remaining_masked - keep_masked)
                    conf_masked[~is_mask] = float("inf")  # revealed stay revealed
                order = torch.argsort(conf_masked, descending=True)
                reveal_idx = order[:n_reveal]

                new_gen = seq[0, gen_slice].clone()
                for j in reveal_idx:
                    if is_mask[j]:
                        new_gen[j] = pred[j]
                seq[0, gen_slice] = new_gen

                is_last = step == steps - 1
                if emit(step, is_last):
                    text, revealed = render_partial(tok, seq[0, gen_slice], mask_id)
                    yield ("step", {
                        "type": "step", "step": step + 1, "total": steps,
                        "text": text, "revealed": revealed, "gen_len": gen_len,
                    })

            # final: force-resolve any leftover masks to their argmax, decode clean
            final_ids = seq[0, gen_slice].clone()
            leftover = final_ids == mask_id
            if bool(leftover.any()):
                logits = model(input_ids=seq).logits[0, gen_slice]
                argmax = logits.argmax(dim=-1)
                final_ids[leftover] = argmax[leftover]
            final_text = tok.decode(final_ids, skip_special_tokens=True)
            yield ("done", {"type": "done", "text": final_text})


# =============================================================================
# Backend: LLaDA (GSAI-ML/LLaDA-8B-Instruct, official low-confidence semi-AR)
# =============================================================================
def _add_gumbel_noise(logits, temperature, torch):
    """Gumbel-max trick for low-precision-safe categorical sampling (official LLaDA).
    temperature<=0 => no noise (greedy argmax)."""
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel


def _get_num_transfer_tokens(mask_index, steps, torch):
    """Per-step count of tokens to unmask within a block, evenly spread with the
    remainder front-loaded (official LLaDA get_num_transfer_tokens)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : remainder[i]] += 1
    return num


class LladaBackend:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("llada mode requires CUDA (8B bf16 model).")
        self.torch = torch
        self.device = "cuda"
        self.mask_id = LLADA_MASK_ID
        self.model_name = LLADA_MODEL

        self.tok = AutoTokenizer.from_pretrained(LLADA_MODEL, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            LLADA_MODEL, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to("cuda").eval()

    def health_model(self) -> str:
        return self.model_name

    def _build_prompt_ids(self, prompt: str):
        """Apply the LLaDA-Instruct chat template, fall back to raw encode."""
        torch = self.torch
        tok = self.tok
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt")["input_ids"]
        except Exception:
            ids = tok(prompt, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def generate_stream(self, prompt: str, gen_len: int, steps: int,
                        block_size: int, temperature: float):
        """Official LLaDA low-confidence semi-autoregressive remasking, yielding
        the partial decode of the answer region each step."""
        torch = self.torch
        import torch.nn.functional as F  # noqa: N812

        model, tok, mask_id, device = self.model, self.tok, self.mask_id, self.device

        with torch.no_grad():
            prompt_ids = self._build_prompt_ids(prompt)
            p = prompt_ids.shape[1]
            total = p + gen_len

            x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
            x[:, :p] = prompt_ids.clone()
            gen_slice = slice(p, total)

            # semi-AR: split the answer region into blocks; resolve left-to-right.
            block_size = max(1, min(block_size, gen_len))
            assert gen_len % block_size == 0 or True  # tolerate non-divisor
            num_blocks = math.ceil(gen_len / block_size)
            steps_per_block = max(1, steps // num_blocks)
            total_steps = steps_per_block * num_blocks

            emit = _emit_steps_filter(total_steps)
            global_step = 0

            for b in range(num_blocks):
                b0 = p + b * block_size
                b1 = min(p + (b + 1) * block_size, total)
                block_mask_index = x[:, b0:b1] == mask_id
                if not bool(block_mask_index.any()):
                    continue
                num_transfer = _get_num_transfer_tokens(
                    block_mask_index, steps_per_block, torch)

                for i in range(steps_per_block):
                    mask_index = x == mask_id
                    logits = model(x).logits
                    logits_noised = _add_gumbel_noise(logits, temperature, torch)
                    x0 = torch.argmax(logits_noised, dim=-1)  # (1, total)

                    # confidence = prob of the chosen token (softmax over real logits)
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)), -1)

                    # only consider currently-masked positions inside THIS block
                    confidence = torch.where(mask_index, x0_p,
                                             torch.tensor(-float("inf"),
                                                          device=device,
                                                          dtype=x0_p.dtype))
                    confidence[:, :b0] = -float("inf")
                    confidence[:, b1:] = -float("inf")

                    x0 = torch.where(mask_index, x0, x)

                    # select the top-k highest-confidence positions to commit
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool,
                                                      device=device)
                    k = int(num_transfer[0, i].item())
                    if k > 0:
                        avail = int((confidence[0] > -float("inf")).sum().item())
                        k = min(k, avail)
                        if k > 0:
                            _, sel = torch.topk(confidence[0], k=k)
                            transfer_index[0, sel] = True
                    x[transfer_index] = x0[transfer_index]

                    is_last = (b == num_blocks - 1) and (i == steps_per_block - 1)
                    if emit(global_step, is_last):
                        text, revealed = render_partial(
                            tok, x[0, gen_slice], mask_id)
                        yield ("step", {
                            "type": "step", "step": global_step + 1,
                            "total": total_steps, "text": text,
                            "revealed": revealed, "gen_len": gen_len,
                        })
                    global_step += 1

            # final clean decode of the answer region (resolve any leftover mask)
            final_ids = x[0, gen_slice].clone()
            leftover = final_ids == mask_id
            if bool(leftover.any()):
                logits = model(x).logits[0, gen_slice]
                argmax = logits.argmax(dim=-1)
                final_ids[leftover] = argmax[leftover]
            final_text = tok.decode(final_ids, skip_special_tokens=True)
            yield ("done", {"type": "done", "text": final_text})


# =============================================================================
# Loading
# =============================================================================
def load_backend(mode: str, base: str, ckpt: str, block_size: int):
    if mode == "berlin":
        backend = BerlinBackend(base=base, ckpt=ckpt, block_size=block_size)
    elif mode == "llada":
        backend = LladaBackend()
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'berlin' or 'llada')")
    _STATE["mode"] = mode
    _STATE["backend"] = backend
    _STATE["model_name"] = backend.health_model()
    print(f"[serve] mode={mode} model={_STATE['model_name']} ready", flush=True)
    return backend


# =============================================================================
# HTTP / SSE
# =============================================================================
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # quieter, structured-ish access log
    def log_message(self, fmt, *args):  # noqa: D401
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    def _set_cors(self):
        for k, v in _CORS_HEADERS.items():
            self.send_header(k, v)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._serve_html()
            return
        if self.path.split("?", 1)[0] == "/api/health":
            self._send_json(200, {
                "ok": True,
                "mode": _STATE.get("mode"),
                "model": _STATE.get("model_name"),
            })
            return
        self._send_json(404, {"type": "error", "message": "not found"})

    def _serve_html(self):
        html_path = os.path.join(_HERE, "chat.html")
        try:
            with open(html_path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            msg = (b"<h1>diffusion-chat backend</h1>"
                   b"<p>chat.html not found next to app.py. The API is up at "
                   b"<code>/api/health</code> and <code>/api/chat</code>.</p>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/chat":
            self._send_json(404, {"type": "error", "message": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:  # noqa: BLE001
            self._send_json(400, {"type": "error", "message": f"bad request: {e}"})
            return

        prompt = str(req.get("prompt", ""))
        try:
            gen_len = int(req.get("gen_len", 128))
            steps = int(req.get("steps", 128))
            block_size = int(req.get("block_size", 32))
            temperature = float(req.get("temperature", 0.3))
        except (TypeError, ValueError) as e:
            self._send_json(400, {"type": "error", "message": f"bad params: {e}"})
            return

        # clamp to sane bounds (avoid OOM / pathological loops)
        gen_len = max(1, min(gen_len, 1024))
        steps = max(1, min(steps, 1024))
        block_size = max(1, min(block_size, gen_len))
        temperature = max(0.0, min(temperature, 5.0))

        # open the SSE stream
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
        self._set_cors()
        self.end_headers()

        def sse(obj: dict):
            chunk = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()  # flush each SSE event immediately (no buffering)

        backend = _STATE.get("backend")
        if backend is None:
            try:
                sse({"type": "error", "message": "model not loaded"})
            except Exception:  # noqa: BLE001
                pass
            return

        # one generation at a time
        if not _GEN_LOCK.acquire(blocking=False):
            try:
                sse({"type": "error",
                     "message": "server busy: another generation is running"})
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            for kind, payload in backend.generate_stream(
                    prompt=prompt, gen_len=gen_len, steps=steps,
                    block_size=block_size, temperature=temperature):
                sse(payload)
                if kind == "done":
                    break
        except BrokenPipeError:
            # client disconnected mid-stream; nothing to do
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                sse({"type": "error", "message": str(e)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            _GEN_LOCK.release()


# =============================================================================
# main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="diffusion-chat backend (SSE)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "berlin"),
                    choices=["berlin", "llada"])
    ap.add_argument("--ckpt", default=os.environ.get(
        "BERLIN_CKPT", "checkpoints/qwen_long/step7400"),
        help="berlin mode: trained checkpoint dir (safetensors + tokenizer), or "
             "'latest' to resolve the newest checkpoints/qwen_long/stepN dir")
    ap.add_argument("--base", default=os.environ.get("BERLIN_BASE", DEFAULT_BASE),
                    help="berlin mode: base model to reconstruct arch from")
    ap.add_argument("--block-size", type=int,
                    default=int(os.environ.get("BLOCK_SIZE", "64")),
                    help="berlin mode: attention block size for surgery")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8890")))
    args = ap.parse_args()

    # resolve the checkpoint spec (handles BERLIN_CKPT=latest -> newest stepN dir).
    ckpt = args.ckpt
    if args.mode == "berlin":
        ckpt = resolve_latest_ckpt(ckpt)
        # resolve ckpt relative to repo root if not absolute / not existing as given
        if ckpt and not os.path.isabs(ckpt) and not os.path.isdir(ckpt):
            cand = os.path.join(_REPO, ckpt)
            if os.path.isdir(cand):
                ckpt = cand

    print(f"[serve] loading mode={args.mode} ...", flush=True)
    load_backend(mode=args.mode, base=args.base, ckpt=ckpt,
                 block_size=args.block_size)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}  "
          f"(GET / , GET /api/health , POST /api/chat)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
