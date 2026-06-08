"""Evaluation harness — stop flying blind on quality.

Computes diffusion perplexity proxy on a held-out German set: for a sweep of mask
ratios t, mask tokens, predict, measure CE on masked tokens; report per-t CE and
an aggregate pseudo-perplexity (exp of mask-averaged CE). Lower = better. Runs on
the base AR model too (baseline) so you can see the surgery+training actually helps.

  python src/diffusion/eval.py --ckpt <hf-repo-or-dir> --base utter-project/EuroLLM-1.7B \
      --eval_jsonl data/eval_de.jsonl --n 200
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


@torch.no_grad()
def eval_perplexity(model, tok, mask_id, texts, device, max_length=512,
                    t_grid=(0.15, 0.3, 0.5, 0.7), block_size=64):
    model.eval()
    per_t = {}
    tot_ce = tot_tok = 0.0
    for t_val in t_grid:
        ce_sum = n = 0.0
        for txt in texts:
            ids = tok(txt, truncation=True, max_length=max_length,
                      return_tensors="pt")["input_ids"].to(device)
            if ids.numel() < 2:
                continue
            t = torch.full((1,), t_val, device=device)
            noised, mb = forward_mask(ids, t, mask_id, block_size=block_size)
            if not mb.any():
                continue
            logits = model(input_ids=noised).logits
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 ids.view(-1), reduction="none").view_as(ids)
            ce_sum += ce[mb].sum().item()
            n += mb.sum().item()
        mean_ce = ce_sum / max(n, 1)
        per_t[t_val] = {"ce": round(mean_ce, 4), "ppl": round(math.exp(mean_ce), 2)}
        tot_ce += ce_sum
        tot_tok += n
    agg_ce = tot_ce / max(tot_tok, 1)
    return {"per_t": per_t, "agg_ce": round(agg_ce, 4),
            "agg_ppl": round(math.exp(agg_ce), 2)}


def load_weights(model, ckpt):
    shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if shards:
        from safetensors.torch import load_file
        state = {}
        for s in shards:
            state.update(load_file(s))
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}  # compile prefix
        missing, _ = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) * 0.5:
            raise SystemExit("[eval] checkpoint failed to load (>50% keys missing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="local ckpt dir OR HF repo id")
    ap.add_argument("--base", default="utter-project/EuroLLM-1.7B")
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--block_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    texts = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                texts.append(json.loads(ln).get("text", ""))
            if len(texts) >= args.n:
                break

    src = args.ckpt if (args.ckpt and not os.path.isdir(args.ckpt)) else args.base
    tok = AutoTokenizer.from_pretrained(args.ckpt if args.ckpt and not os.path.isdir(args.ckpt) else args.base)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16 if device == "cuda" else None)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    if args.ckpt and os.path.isdir(args.ckpt):
        load_weights(model, args.ckpt)
    model.to(device)
    apply_blockwise_bidirectional(model, block_size=args.block_size)
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)

    res = eval_perplexity(model, tok, mask_id, texts, device, block_size=args.block_size)
    print(f"[eval] n={len(texts)} src={src}")
    for t_val, m in res["per_t"].items():
        print(f"[eval]  t={t_val:.2f}  CE={m['ce']:.4f}  ppl={m['ppl']:.2f}")
    print(f"[eval]  AGG  CE={res['agg_ce']:.4f}  pseudo-ppl={res['agg_ppl']:.2f}")


if __name__ == "__main__":
    main()
