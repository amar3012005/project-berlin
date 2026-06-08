"""Save a trained Project Berlin checkpoint to the HuggingFace Hub.

The checkpoint (3.3GB safetensors) is too big for git. Pushing it to a private
HF repo is the clean hand-off: any RunPod / cloud instance pulls it with one line.
Also writes config.json so it reloads as a normal AutoModel.

  HF_TOKEN=hf_xxx python scripts/push_to_hub.py \
      --ckpt checkpoints/eurollm_cluster/final \
      --repo <user>/project-berlin-eurollm-de --base utter-project/EuroLLM-1.7B
"""
from __future__ import annotations

import argparse
import glob
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="target HF repo id, e.g. user/name")
    ap.add_argument("--base", default="utter-project/EuroLLM-1.7B")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok), mean_resizing=False)

    shards = sorted(glob.glob(os.path.join(args.ckpt, "*.safetensors")))
    state = {}
    for s in shards:
        state.update(load_file(s))
    # strip torch.compile's "_orig_mod." prefix (compiled-run checkpoints) else keys
    # mismatch and load_state_dict(strict=False) silently loads NOTHING
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[push] merged {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > len(state) * 0.5:
        raise SystemExit("[push] ABORT: >50% keys missing — checkpoint didn't load, refusing to push base model")

    print(f"[push] uploading -> https://huggingface.co/{args.repo} (private={args.private})")
    model.push_to_hub(args.repo, private=args.private)
    tok.push_to_hub(args.repo, private=args.private)
    print("[push] DONE. Pull on RunPod with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{args.repo}')")


if __name__ == "__main__":
    main()
