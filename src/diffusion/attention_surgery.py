"""M1: Attention rewiring — causal -> block-wise bidirectional (ARCHITECTURE-AGNOSTIC).

Per "Scaling Diffusion Language Models via Adaptation from AR Models" (ICLR 2025)
and LLaDA (2025). An AR model attends only backward (causal mask). A discrete-
diffusion model must denoise a noisy block with FULL context: forward AND backward
within the noisy block, plus full attention to previous clean blocks.

Block-wise bidirectional rule (block size B, block(i) = i // B):
    query q attends key k  iff  block(k) <= block(q)

GENERIC: transformers 5.x routes every decoder (gpt2, llama, mistral, Pharia-1...)
through the SAME `create_causal_mask` imported into each model module. We patch the
function in the running model's own module namespace, so the identical surgery works
on the 124M german-gpt2 toy AND on Aleph-Alpha/Pharia-1-LLM-7B-control (llama-arch)
with NO code change — only the config's model name differs. Reversible via restore().
"""
from __future__ import annotations

import importlib

import torch

# remember originals we patch so restore() is exact, keyed by module path
_PATCHED: dict[str, object] = {}


def build_blockwise_bidirectional_mask(
    seq_len: int,
    block_size: int,
    device,
    dtype,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive 4D mask (batch?,1,q,kv): 0.0 = attend, -inf = block."""
    idx = torch.arange(seq_len, device=device)
    blk = idx // block_size
    allowed = blk[None, :] <= blk[:, None]
    neg = torch.finfo(dtype).min
    mask = torch.where(allowed, 0.0, neg).to(dtype)
    mask = mask[None, None, :, :]
    if padding_mask is not None:
        pad = (1 - padding_mask[:, None, None, :].to(dtype)) * neg
        mask = mask + pad
    return mask


def _model_module(model):
    """Module object where the model class (and its create_causal_mask) lives."""
    return importlib.import_module(type(model).__module__)


def apply_blockwise_bidirectional(model, block_size: int = 64) -> None:
    """Patch the model's own module-level create_causal_mask. Works for any
    transformers decoder arch (gpt2 / llama / mistral / Pharia-1)."""
    mod = _model_module(model)
    if not hasattr(mod, "create_causal_mask"):
        raise RuntimeError(
            f"{mod.__name__} has no create_causal_mask — unsupported transformers "
            "version or arch; inspect the model's *Model.forward mask call site."
        )
    key = mod.__name__
    if key not in _PATCHED:
        _PATCHED[key] = (mod, mod.create_causal_mask)

    def _patched(config, inputs_embeds, attention_mask, past_key_values,
                 position_ids=None, **kwargs):
        seq_len = inputs_embeds.shape[1]
        return build_blockwise_bidirectional_mask(
            seq_len=seq_len,
            block_size=block_size,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            padding_mask=attention_mask if (attention_mask is not None
                                            and attention_mask.ndim == 2) else None,
        )

    mod.create_causal_mask = _patched
    # eager honours arbitrary float masks on every backend incl. CPU/MPS.
    # On GPU you may instead use 'sdpa' (set in config) — float mask still works.
    if getattr(model.config, "_attn_implementation", None) in (None, "flash_attention_2"):
        model.config._attn_implementation = "eager"
    model._berlin_block_size = block_size


def restore(model=None) -> None:
    for key, (mod, orig) in list(_PATCHED.items()):
        mod.create_causal_mask = orig
        del _PATCHED[key]
    if model is not None and hasattr(model, "_berlin_block_size"):
        del model._berlin_block_size


def _selftest() -> None:
    from load_model import load

    model, tok, device = load()
    block = 4
    apply_blockwise_bidirectional(model, block_size=block)

    m = build_blockwise_bidirectional_mask(8, block, torch.device(device), torch.float32)
    m2 = m[0, 0]
    neg = torch.finfo(torch.float32).min
    assert m2[0, 3] == 0.0, "within-block forward attention must be allowed"
    assert m2[0, 4] == neg, "future block must be masked"
    assert m2[4, 0] == 0.0, "past block must be visible"
    print("[M1] mask semantics OK  (intra-block bidirectional, block-causal across)")
    print(f"[M1] patched module: {list(_PATCHED)[0]}")

    ids = tok("Die deutsche Sprache ist sehr alt und schoen", return_tensors="pt").to(device)
    seq = ids["input_ids"]
    with torch.no_grad():
        base = model(input_ids=seq).logits[0, 0].clone()
        seq2 = seq.clone()
        seq2[0, 2] = seq2[0, 3]
        pert = model(input_ids=seq2).logits[0, 0]
    delta = (base - pert).abs().max().item()
    print(f"[M1] logits[pos0] change after perturbing future token pos2: {delta:.4f}")
    assert delta > 1e-4, "bidirectional attention should let future tokens affect pos0"
    print("[M1] behavioural proof OK — attention is now bidirectional")
    restore(model)
    print("[M1] restored. OK")


if __name__ == "__main__":
    _selftest()
