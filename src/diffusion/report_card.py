"""Capability report card — run at EVERY training stage to track progress.

Fixed German prompts → long paragraph generation (diffusion reverse process) +
pseudo-perplexity. Same inputs every time, so you can eyeball quality climbing as
tokens_seen grows. Writes a timestamped card to reports/ for side-by-side comparison.

  python src/diffusion/report_card.py --ckpt checkpoints/eurollm_cluster/step1000 \
      --base utter-project/EuroLLM-1.7B --gen_len 160 --steps 24
  # or a published model:
  python src/diffusion/report_card.py --ckpt amar3012005/project-berlin-eurollm-de
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_surgery import apply_blockwise_bidirectional
from diffusion import MASK_TOKEN, forward_mask
from generate import generate

# Fixed test set — legal / technical / general German registers. Never change these
# (so stages are comparable). Each is a realistic "similar input" for the domain.
PROMPTS = [
    "Das Gericht entschied, dass der Angeklagte",
    "Gemäß DIN-Norm muss die elektrische Anlage",
    "Die deutsche Wirtschaft entwickelte sich im letzten Jahr",
    "Der Vertrag zwischen den beiden Unternehmen regelt",
    "Künstliche Intelligenz verändert die Arbeitswelt, weil",
]


def load_model(ckpt, base, device):
    is_local = os.path.isdir(ckpt)
    src = base if is_local else ckpt
    tok = AutoTokenizer.from_pretrained(ckpt if not is_local else base)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16 if device == "cuda" else None)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    if is_local:
        from safetensors.torch import load_file
        state = {}
        for s in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
            state.update(load_file(s))
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}  # compile prefix
        missing, _ = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) * 0.5:
            print(f"  !! WARNING: {len(missing)}/{len(state)} keys missing — "
                  "near-base model, NOT the trained checkpoint")
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def quick_ppl(model, tok, mask_id, device, block_size):
    texts = ["Die Vorschrift nach DIN 18065 regelt die Treppenmasse im Bauwesen "
             "und legt die zulaessigen Steigungen fest."] * 8
    ce_sum = n = 0.0
    for t_val in (0.3, 0.5, 0.7):
        for txt in texts[:3]:
            ids = tok(txt, return_tensors="pt")["input_ids"].to(device)
            t = torch.full((1,), t_val, device=device)
            noised, mb = forward_mask(ids, t, mask_id, block_size=block_size)
            if not mb.any():
                continue
            logits = model(input_ids=noised).logits
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 ids.view(-1), reduction="none").view_as(ids)
            ce_sum += ce[mb].sum().item(); n += mb.sum().item()
    ce = ce_sum / max(n, 1)
    return round(ce, 4), round(math.exp(ce), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="utter-project/EuroLLM-1.7B")
    ap.add_argument("--gen_len", type=int, default=160)   # big paragraph
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--stage", default="")                # label, e.g. "step1000"
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model, tok = load_model(args.ckpt, args.base, device)
    apply_blockwise_bidirectional(model, block_size=args.block_size)
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)

    ce, ppl = quick_ppl(model, tok, mask_id, device, args.block_size)
    label = args.stage or os.path.basename(args.ckpt.rstrip("/"))
    lines = [f"# Project Berlin — Report Card [{label}]",
             f"ckpt={args.ckpt}  device={device}  gen_len={args.gen_len} steps={args.steps}",
             f"pseudo-perplexity={ppl}  (CE={ce}) — lower=better; ~1000=untrained, tens=good",
             ""]
    print("\n".join(lines))

    for p in PROMPTS:
        out = generate(model, tok, mask_id, prompt=p, gen_len=args.gen_len,
                       steps=args.steps, device=device, temperature=args.temperature,
                       repetition_penalty=1.3, conf_threshold=0.0)
        block = f"PROMPT: {p}\nOUTPUT: {p} {out}\n" + "-" * 70
        print(block)
        lines.append(block)

    os.makedirs("reports", exist_ok=True)
    fn = f"reports/card_{label}.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[card] saved -> {fn}")


if __name__ == "__main__":
    main()
