"""M2: Discrete diffusion forward masking + denoising loss (LLaDA, Feb 2025).

Forward process: sample t ~ U[0,1]; each token is independently replaced by
[MASK] with probability t. The model (bidirectional, post-surgery) sees the noisy
sequence and predicts the original tokens. Loss = cross-entropy on the MASKED
positions only, weighted by 1/t (Monte-Carlo estimate of the NELBO bound, eq. in
LLaDA / SEDD).

german-gpt2 has no mask token, so we add one and resize the embedding matrix —
the toy-scale equivalent of reserving a [MASK] id in the Pharia-1 tokenizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_TOKEN = "[MASK]"


def add_mask_token(model, tok) -> int:
    """Add [MASK] to the tokenizer + resize embeddings. Returns mask token id."""
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
        model.resize_token_embeddings(len(tok))
    return tok.convert_tokens_to_ids(MASK_TOKEN)


def forward_mask(input_ids: torch.Tensor, t: torch.Tensor, mask_id: int):
    """Noise the sequence. t: (batch,) masking prob per sequence.
    Returns (noised_ids, mask_bool) where mask_bool marks corrupted positions."""
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_bool = rand < t[:, None]                       # (batch, seq)
    noised = torch.where(mask_bool, mask_id, input_ids)
    return noised, mask_bool


def denoise_loss(model, input_ids: torch.Tensor, mask_id: int,
                 t_min: float = 1e-3):
    """One diffusion training step's loss. Samples t, masks, predicts originals."""
    batch = input_ids.shape[0]
    t = torch.rand(batch, device=input_ids.device).clamp_min(t_min)
    noised, mask_bool = forward_mask(input_ids, t, mask_id)

    # force >=1 masked token per row so the loss is always defined
    no_mask = ~mask_bool.any(dim=1)
    if no_mask.any():
        first = torch.zeros_like(mask_bool)
        first[no_mask, 0] = True
        mask_bool = mask_bool | first
        noised = torch.where(mask_bool, mask_id, input_ids)

    logits = model(input_ids=noised).logits          # (batch, seq, vocab)

    # per-token CE on masked positions only
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        reduction="none",
    ).view_as(input_ids)                              # (batch, seq)

    # 1/t weighting (LLaDA NELBO estimator), averaged over masked tokens
    weight = (1.0 / t)[:, None]
    masked_ce = ce * mask_bool.float() * weight
    loss = masked_ce.sum() / mask_bool.float().sum().clamp_min(1.0)
    return loss, mask_bool


def _selftest() -> None:
    from load_model import load
    from attention_surgery import apply_blockwise_bidirectional

    model, tok, device = load()
    apply_blockwise_bidirectional(model, block_size=64)
    mask_id = add_mask_token(model, tok)
    print(f"[M2] added {MASK_TOKEN} id={mask_id} vocab={len(tok)}")

    text = ["Die Vorschrift nach DIN 18065 regelt die Treppenmasse im Bauwesen."]
    ids = tok(text, return_tensors="pt").to(device)["input_ids"]

    # forward-mask sanity at t=0.5
    t = torch.full((1,), 0.5, device=device)
    noised, mb = forward_mask(ids, t, mask_id)
    frac = mb.float().mean().item()
    print(f"[M2] t=0.5 masked fraction={frac:.2f} (expect ~0.5)")
    assert (noised[mb] == mask_id).all(), "masked positions must hold mask id"
    assert (noised[~mb] == ids[~mb]).all(), "unmasked positions must be untouched"

    # loss computes + backprops
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss, _ = denoise_loss(model, ids, mask_id)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    opt.step()
    print(f"[M2] denoise loss={loss.item():.4f}  grad_norm={gnorm:.4f}")
    assert loss.item() > 0 and gnorm > 0, "loss must be positive and produce gradients"
    print("[M2] OK — forward masking + denoise loss backprops")


if __name__ == "__main__":
    _selftest()
