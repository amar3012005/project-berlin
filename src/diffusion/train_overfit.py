"""M4: Overfit a tiny German corpus to prove the diffusion training loop learns.

This is the end-to-end validation gate before renting GPUs: load -> surgery ->
forward-mask -> denoise loss -> backprop, repeated. If the loss falls and the
model can reconstruct masked tokens of a memorised sentence, the full pipeline
(M0-M3 wired together) is correct and ready to scale to Pharia-1-7B on
ThunderCompute.
"""
from __future__ import annotations

import torch

from load_model import load
from attention_surgery import apply_blockwise_bidirectional
from diffusion import add_mask_token, denoise_loss, forward_mask

CORPUS = [
    "Die Vorschrift nach DIN 18065 regelt die Treppenmasse im Bauwesen.",
    "Der Ingenieur prueft die elektrische Anlage gemaess VDE-Norm.",
    "Die deutsche Sprache verwendet zusammengesetzte Hauptwoerter.",
    "Das Unternehmen erfuellt alle Anforderungen der Qualitaetssicherung.",
    "Die Mitarbeiter erhalten eine Schulung zur Arbeitssicherheit.",
    "Der Vertrag wird nach deutschem Recht geschlossen und ausgelegt.",
]


@torch.no_grad()
def reconstruction_accuracy(model, tok, mask_id, device, mask_frac=0.4):
    """Mask a fixed fraction of each sentence, measure token recovery accuracy."""
    model.eval()
    correct = total = 0
    for text in CORPUS:
        ids = tok(text, return_tensors="pt").to(device)["input_ids"]
        t = torch.full((1,), mask_frac, device=device)
        noised, mb = forward_mask(ids, t, mask_id)
        if not mb.any():
            continue
        pred = model(input_ids=noised).logits.argmax(-1)
        correct += (pred[mb] == ids[mb]).sum().item()
        total += mb.sum().item()
    return correct / max(total, 1)


def main(steps: int = 300, lr: float = 5e-5, log_every: int = 50) -> None:
    model, tok, device = load()
    apply_blockwise_bidirectional(model, block_size=64)
    mask_id = add_mask_token(model, tok)

    batch = [tok(t, return_tensors="pt").to(device)["input_ids"] for t in CORPUS]

    acc0 = reconstruction_accuracy(model, tok, mask_id, device)
    print(f"[M4] reconstruction acc BEFORE training: {acc0:.3f}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(0)
    first_loss = None
    for step in range(1, steps + 1):
        ids = batch[step % len(batch)]
        loss, _ = denoise_loss(model, ids, mask_id)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % log_every == 0 or step == 1:
            print(f"[M4] step {step:4d}  loss={loss.item():8.4f}")

    acc1 = reconstruction_accuracy(model, tok, mask_id, device)
    print(f"[M4] reconstruction acc AFTER  training: {acc1:.3f}")
    print(f"[M4] loss {first_loss:.2f} -> {loss.item():.2f}  |  acc {acc0:.3f} -> {acc1:.3f}")
    assert loss.item() < first_loss, "loss must drop — training not learning"
    assert acc1 > acc0, "reconstruction accuracy must improve"
    print("[M4] OK — diffusion training loop learns. Pipeline ready to scale.")


if __name__ == "__main__":
    main()
