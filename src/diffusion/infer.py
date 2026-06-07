"""Inference + LIVE diffusion reveal viewer.

Loads a trained Project Berlin checkpoint and generates German text via reverse
diffusion, printing the partial unmasking each step (the 'Inception' effect:
masked slots ▒ fill into real German token-by-token).

  python src/diffusion/infer.py --ckpt checkpoints/eurollm_real/final \
      --prompt "Das Gericht entschied" --gen_len 48 --steps 16 --show_steps
"""
from __future__ import annotations

import argparse

import glob
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_surgery import apply_blockwise_bidirectional
from diffusion import MASK_TOKEN
from generate import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint dir")
    ap.add_argument("--base", default="utter-project/EuroLLM-1.7B",
                    help="base model to reconstruct arch from (ckpt holds weights only)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--gen_len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--show_steps", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    # tokenizer (with [MASK]) is saved in the ckpt; arch comes from base + resize
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16 if device == "cuda" else None)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    # load trained weights (accel.save_model writes safetensors shard[s], no config)
    shards = sorted(glob.glob(os.path.join(args.ckpt, "*.safetensors")))
    state = {}
    for s in shards:
        state.update(load_file(s))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[infer] loaded {len(state)} tensors from {len(shards)} shard(s); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
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
