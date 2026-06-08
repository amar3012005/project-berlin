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
    dtype,  # kept for signature compat; boolean mask is dtype-agnostic
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean 4D mask (batch?,1,q,kv): True = attend, False = block.

    Block-wise bidirectional: block(k) <= block(q). We pass this BOOLEAN mask
    straight to SDPA (True=keep). SDPA sees a non-None mask -> sets is_causal=False
    internally -> dense within-and-prior blocks. The diagonal is forced True so every
    query row always has >=1 visible key => SDPA/flash never sees a fully-masked row
    => no NaN/hang in the backward pass (the finfo.min footgun that hung autograd
    via the all-masked-row softmax-gradient NaN, pytorch #110213, is gone)."""
    idx = torch.arange(seq_len, device=device)
    blk = idx // block_size
    allowed = blk[None, :] <= blk[:, None]                 # (q, kv) bool
    allowed = allowed | torch.eye(seq_len, dtype=torch.bool, device=device)
    mask = allowed[None, None, :, :]                       # (1,1,q,kv)
    if padding_mask is not None:
        pad_keep = padding_mask[:, None, None, :].to(torch.bool)   # (b,1,1,kv)
        mask = mask & pad_keep                             # (b,1,q,kv)
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None, :, :]
        mask = mask | eye                                  # re-guarantee >=1 visible key
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
    # Do NOT force eager. SDPA natively consumes the BOOLEAN mask (True=keep) and,
    # because the mask is non-None, sets is_causal=False -> block-wise bidirectional
    # with ZERO additive -inf. eager + additive finfo.min was the backward-hang
    # trigger (all-masked-row softmax-grad NaN, pytorch #110213); it is gone.
    # flash kernel can't take an arbitrary boolean attn_mask -> route to sdpa.
    if getattr(model.config, "_attn_implementation", None) == "flash_attention_2":
        model.config._attn_implementation = "sdpa"
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
    assert bool(m2[0, 3]), "within-block forward attention must be allowed (True=attend)"
    assert not bool(m2[0, 4]), "future block must be masked (False)"
    assert bool(m2[4, 0]), "past block must be visible (True)"
    assert bool(m2.diagonal().all()), "diagonal must always be True (no fully-masked row)"
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
