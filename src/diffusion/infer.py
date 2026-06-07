"""Inference + LIVE diffusion reveal viewer.

Loads a trained Project Berlin checkpoint and generates German text via reverse
diffusion, printing the partial unmasking each step (the 'Inception' effect:
masked slots ▒ fill into real German token-by-token).

  python src/diffusion/infer.py --ckpt checkpoints/eurollm_real/final \
      --prompt "Das Gericht entschied" --gen_len 48 --steps 16 --show_steps
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_surgery import apply_blockwise_bidirectional
from diffusion import MASK_TOKEN
from generate import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint dir")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--gen_len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--show_steps", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16 if device == "cuda" else None)
    model.to(device).eval()
    apply_blockwise_bidirectional(model, block_size=args.block_size)
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)

    print(f"=== prompt: {args.prompt!r} | gen_len={args.gen_len} steps={args.steps} ===")
    out = generate(model, tok, mask_id, prompt=args.prompt, gen_len=args.gen_len,
                   steps=args.steps, device=device, temperature=args.temperature,
                   show_steps=args.show_steps)
    print("\n=== FINAL ===")
    print(args.prompt + out)


if __name__ == "__main__":
    main()
