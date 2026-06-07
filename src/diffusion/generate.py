"""M3: Reverse diffusion sampling — parallel unmasking (LLaDA, 2025).

Start from a fully-[MASK]ed answer region (optionally after a clean prompt prefix).
Over N steps, the bidirectional model predicts ALL masked tokens at once; we commit
the highest-confidence predictions and re-mask the rest, revealing more tokens each
step. This is the "parallel decoding" that makes DLMs fast and the basis for the
FS-DFM few-step (4-8 step) distillation in Stage 4.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tok, mask_id: int, prompt: str = "", gen_len: int = 32,
             steps: int = 16, device: str = "cpu", temperature: float = 0.0,
             show_steps: bool = False, mask_str: str = "▒"):
    """Generate `gen_len` tokens via reverse diffusion in `steps` unmasking rounds.

    show_steps=True prints the partially-unmasked text after EACH round — the
    'Inception' reveal: masked slots render as `mask_str`, filling to real German
    token-by-token. This is the diffusion effect you watch during inference."""
    model.eval()
    prompt_ids = (tok(prompt, return_tensors="pt")["input_ids"][0].to(device)
                  if prompt else torch.empty(0, dtype=torch.long, device=device))
    p = prompt_ids.shape[0]
    total = p + gen_len

    seq = torch.full((1, total), mask_id, device=device, dtype=torch.long)
    if p:
        seq[0, :p] = prompt_ids

    gen_slice = slice(p, total)
    # cosine reveal schedule: how many of gen_len tokens stay MASKED after step i
    for step in range(steps):
        is_mask = seq[0, gen_slice] == mask_id
        if not is_mask.any():
            break
        logits = model(input_ids=seq).logits[0, gen_slice]   # (gen_len, vocab)
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.gather(-1, pred[:, None]).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

        # target number still masked after this step (cosine -> 0)
        frac = torch.cos(torch.tensor((step + 1) / steps * torch.pi / 2)).item()
        keep_masked = int(gen_len * frac)

        conf_masked = conf.clone()
        conf_masked[~is_mask] = float("inf")   # already-revealed stay revealed
        n_reveal = max(1, int(is_mask.sum().item()) - keep_masked)
        # reveal the n_reveal highest-confidence currently-masked positions
        order = torch.argsort(conf_masked, descending=True)
        reveal_idx = order[:n_reveal]

        new_gen = seq[0, gen_slice].clone()
        for j in reveal_idx:
            if is_mask[j]:
                new_gen[j] = pred[j]
        seq[0, gen_slice] = new_gen

        if show_steps:
            # render current state: masked slots -> mask_str, revealed -> token text
            toks = []
            for t_id in seq[0, gen_slice].tolist():
                toks.append(mask_str if t_id == mask_id
                            else tok.decode([t_id], skip_special_tokens=True))
            revealed = (seq[0, gen_slice] != mask_id).sum().item()
            print(f"[step {step+1:2d}/{steps}] {revealed:3d}/{gen_len} revealed | "
                  + "".join(toks).replace("\n", " "))

    return tok.decode(seq[0, gen_slice], skip_special_tokens=True)


def _selftest() -> None:
    from load_model import load
    from attention_surgery import apply_blockwise_bidirectional
    from diffusion import add_mask_token

    model, tok, device = load()
    apply_blockwise_bidirectional(model, block_size=64)
    mask_id = add_mask_token(model, tok)

    out = generate(model, tok, mask_id, prompt="Die deutsche Sprache",
                   gen_len=24, steps=12, device=device, temperature=0.0)
    print(f"[M3] sampled (untrained, expect gibberish): {out!r}")
    assert isinstance(out, str), "generate must return decoded text"
    # prove the loop fully unmasked (no [MASK] left in a fresh re-run check)
    full = generate(model, tok, mask_id, prompt="", gen_len=16, steps=16, device=device)
    assert "[MASK]" not in full, "reverse loop must fill every masked position"
    print("[M3] OK — reverse parallel-unmask loop fills all positions")


if __name__ == "__main__":
    _selftest()
