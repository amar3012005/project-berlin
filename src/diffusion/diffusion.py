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
        # mean_resizing=False: we add ONE token; the multivariate-normal covariance
        # init costs ~5min on a 128k-vocab model for zero benefit. Mean init is instant.
        model.resize_token_embeddings(len(tok), mean_resizing=False)
    return tok.convert_tokens_to_ids(MASK_TOKEN)


def sample_t(batch: int, device, schedule: str = "uniform",
             t_min: float = 0.05, t_max: float = 1.0,
             beta_a: float = 2.0, beta_b: float = 5.0) -> torch.Tensor:
    """Sample the per-sequence mask ratio t.

    schedule:
      uniform : t ~ U[t_min, t_max]            (LLaDA default)
      beta    : t ~ Beta(a,b) (skewed low)     (Dream-7B/CART: more mass on small
                masks => easy-correction signal; better perplexity/reasoning)
    """
    if schedule == "beta":
        d = torch.distributions.Beta(beta_a, beta_b)
        t = d.sample((batch,)).to(device)
    else:
        t = torch.rand(batch, device=device)
    return (t * (t_max - t_min) + t_min).clamp(t_min, t_max)


def forward_mask(input_ids: torch.Tensor, t: torch.Tensor, mask_id: int,
                 block_size: int = 0):
    """Noise the sequence. t: (batch,) masking prob per sequence.

    block_size>0 => BLOCK-ALIGNED masking: decide mask per `block_size`-token block
    (aligned with the bidirectional attention blocks, LLaDA-style) instead of fully
    independent per-token. Returns (noised_ids, mask_bool)."""
    b, n = input_ids.shape
    if block_size and block_size > 1:
        nblocks = (n + block_size - 1) // block_size
        blk_rand = torch.rand(b, nblocks, device=input_ids.device)
        blk_mask = blk_rand < t[:, None]                       # (b, nblocks)
        mask_bool = blk_mask.repeat_interleave(block_size, dim=1)[:, :n]
    else:
        rand = torch.rand(input_ids.shape, device=input_ids.device)
        mask_bool = rand < t[:, None]
    noised = torch.where(mask_bool, mask_id, input_ids)
    return noised, mask_bool


def denoise_loss(model, input_ids: torch.Tensor, mask_id: int,
                 t_min: float = 0.05, schedule: str = "uniform",
                 block_size: int = 0, weight_alpha: float = 0.3,
                 attention_mask: torch.Tensor | None = None):
    """One diffusion training step's loss. Samples t, masks, predicts originals.

    Loss weight w(t) = alpha + (1-alpha)/t  — a convex blend of constant and the
    1/t NELBO estimator. Pure 1/t spikes hard on low-t batches (high variance);
    blending with a constant floor (alpha) tames the variance while keeping the
    NELBO signal (Dream-7B/LLaDA-style smoothing)."""
    batch = input_ids.shape[0]
    t = sample_t(batch, input_ids.device, schedule=schedule, t_min=t_min)
    noised, mask_bool = forward_mask(input_ids, t, mask_id, block_size=block_size)

    # never mask PAD positions (packed/padded inputs); never count them in loss
    if attention_mask is not None:
        pad = attention_mask == 0
        mask_bool = mask_bool & ~pad
        noised = torch.where(mask_bool, mask_id, input_ids)

    # force >=1 masked token per row so the loss is always defined
    no_mask = ~mask_bool.any(dim=1)
    if no_mask.any():
        first = torch.zeros_like(mask_bool)
        first[no_mask, 0] = True
        mask_bool = mask_bool | first
        noised = torch.where(mask_bool, mask_id, input_ids)

    logits = model(input_ids=noised).logits          # (batch, seq, vocab)

    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),     # CE in fp32 for bf16 stability
        input_ids.view(-1),
        reduction="none",
    ).view_as(input_ids)                              # (batch, seq)

    # NELBO Monte-Carlo weight w(t) = alpha + (1-alpha)/t  (alpha=0 => pure 1/t LLaDA).
    # t_min already floors t, so 1/t is bounded; clamp inv_t defensively so a single
    # tiny-t batch can never emit an inf/NaN gradient (the t_min floor is the real guard).
    inv_t = (1.0 / t.clamp_min(t_min)).clamp_max(1.0 / max(t_min, 1e-3))
    weight = (weight_alpha + (1.0 - weight_alpha) * inv_t)[:, None]   # (batch,1)

    mask_f = mask_bool.float()
    masked_ce = ce * mask_f * weight
    loss = masked_ce.sum() / mask_f.sum().clamp_min(1.0)   # DiffuLLaMA: /#masked, >=1
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
