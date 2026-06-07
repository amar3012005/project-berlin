"""M0: Load small German AR model, prove forward + generation works.

Toy stand-in for Aleph-Alpha/Pharia-1-LLM-7B-control. Uses dbmdz/german-gpt2
(124M) so the full diffusion-surgery pipeline can be proven end-to-end on CPU/MPS
at zero cost before renting ThunderCompute for the real 7B conversion.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "dbmdz/german-gpt2"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tok, device


def load_from_cfg(model_cfg: dict, device_map=None):
    """Config-driven load — same call works for german-gpt2 and Pharia-1-7B.
    On GPU (under accelerate) pass device placement to the trainer, not here."""
    import torch as _t
    name = model_cfg["name"]
    kw = {"trust_remote_code": model_cfg.get("trust_remote_code", False)}
    attn = model_cfg.get("attn_implementation")
    if attn:
        kw["attn_implementation"] = attn
    if device_map is not None:
        kw["device_map"] = device_map
    # bf16 weights on CUDA, fp32 elsewhere
    if _t.cuda.is_available():
        kw["dtype"] = _t.bfloat16
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=kw["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    return model, tok


def main() -> None:
    model, tok, device = load()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M0] loaded {MODEL_NAME} | params={n_params/1e6:.1f}M | device={device}")

    prompt = "Die deutsche Sprache ist"
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    print("[M0] prompt :", prompt)
    print("[M0] greedy :", tok.decode(out[0], skip_special_tokens=True))

    # raw forward pass sanity: logits shape == (batch, seq, vocab)
    with torch.no_grad():
        logits = model(**ids).logits
    print(f"[M0] forward logits shape={tuple(logits.shape)} vocab={model.config.vocab_size}")
    print("[M0] OK")


if __name__ == "__main__":
    main()
