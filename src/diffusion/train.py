"""Project Berlin — production training entrypoint (Stage 2 discrete diffusion).

Runs IDENTICALLY on:
  * laptop CPU/MPS   :  python src/diffusion/train.py configs/toy_gpt2.yaml
  * 8x A100/H100     :  accelerate launch --config_file configs/accelerate_fsdp.yaml \
                           src/diffusion/train.py configs/pharia_7b.yaml

Pipeline (all proven at toy scale, M0-M4): load model -> attention surgery
(causal->block-wise bidirectional) -> add [MASK] -> forward-mask t~U[0,1] ->
cross-entropy on masked tokens (1/t NELBO weight) -> backprop -> checkpoint.

Architecture-agnostic: german-gpt2 (GPT2) and Pharia-1-7B (llama) hit the same
create_causal_mask patch, so swapping models is a one-line YAML change — no rebuild.
"""
from __future__ import annotations

import os
import sys
import json
import math
import yaml

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from load_model import load_from_cfg                       # noqa: E402
from attention_surgery import apply_blockwise_bidirectional  # noqa: E402
from diffusion import add_mask_token, denoise_loss          # noqa: E402
from dataset import build_dataloader                         # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    precision = tcfg.get("precision", "bf16")
    # MPS/CPU can't do bf16 mixed-precision via accelerate -> fall back to no mp
    mp = precision if torch.cuda.is_available() else "no"
    accel = Accelerator(
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        mixed_precision=mp,
        log_with="wandb" if tcfg.get("wandb") else None,
    )
    set_seed(tcfg.get("seed", 0))
    if tcfg.get("wandb") and accel.is_main_process:
        accel.init_trackers(tcfg.get("wandb_project", "project-berlin"),
                            config=cfg)

    accel.print(f"[train] model={mcfg['name']} precision={mp} "
                f"procs={accel.num_processes} device={accel.device}")

    model, tok = load_from_cfg(mcfg)
    apply_blockwise_bidirectional(model, block_size=mcfg.get("block_size", 64))
    mask_id = add_mask_token(model, tok)
    accel.print(f"[train] surgery applied, [MASK] id={mask_id}, vocab={len(tok)}")

    # gradient checkpointing — trade compute for memory => bigger global batch
    if tcfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        accel.print("[train] gradient checkpointing ON")

    # torch.compile — packing gives static shapes => safe + max-autotune. CUDA-only.
    if mcfg.get("compile") and torch.cuda.is_available():
        model = torch.compile(model, mode=mcfg.get("compile_mode", "max-autotune"))
        accel.print(f"[train] torch.compile ({mcfg.get('compile_mode','max-autotune')})")

    loader = build_dataloader(dcfg, tok, tcfg.get("per_device_batch_size", 4),
                              shuffle=not dcfg.get("streaming", False),
                              num_workers=tcfg.get("num_workers", 0))

    optim = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                              weight_decay=float(tcfg.get("weight_decay", 0.0)))

    max_steps = tcfg.get("max_steps")
    epochs = tcfg.get("epochs", 1)
    warmup = tcfg.get("warmup_steps", 0)
    total_steps = max_steps or 100000          # cosine horizon
    min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.1))

    def lr_lambda(step):
        if warmup and step < warmup:
            return step / max(1, warmup)
        # cosine decay warmup->total_steps, floor at min_lr_ratio (better convergence)
        prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    model, optim, loader, sched = accel.prepare(model, optim, loader, sched)

    out_dir = tcfg["output_dir"]
    # RESUME: restore model+optim+sched+RNG so a RunPod interruption costs minutes,
    # not the whole run. accel.save_state/load_state handles all of it.
    start_step = 0
    resume_from = tcfg.get("resume_from") or os.environ.get("BERLIN_RESUME")
    if resume_from and os.path.isdir(resume_from):
        accel.load_state(resume_from)
        sf = os.path.join(resume_from, "berlin_step.txt")
        if os.path.exists(sf):
            with open(sf) as f:
                start_step = int(f.read().strip())
        accel.print(f"[train] RESUMED from {resume_from} at step {start_step}")

    model.train()
    t_min = float(tcfg.get("t_min", 1e-3))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = tcfg.get("log_every", 20)
    save_every = tcfg.get("save_every", 100)
    nan_threshold = float(tcfg.get("nan_loss_threshold", 1e4))

    # realtime metrics file consumed by the monitor dashboard (src/monitor)
    import time
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if accel.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        open(metrics_path, "w").close()                  # reset each run
    seq_len = int(dcfg.get("max_length", 512))
    global_batch = (tcfg.get("per_device_batch_size", 4)
                    * tcfg.get("grad_accum", 1) * accel.num_processes)
    tokens_per_step = global_batch * seq_len
    t_last = time.time()

    step = start_step
    tokens_seen = start_step * tokens_per_step
    done = False
    for epoch in range(epochs):
        if done:
            break
        for batch in loader:
            with accel.accumulate(model):
                loss, _ = denoise_loss(
                    model, batch["input_ids"], mask_id, t_min=t_min,
                    schedule=tcfg.get("mask_schedule", "uniform"),
                    block_size=tcfg.get("mask_block_size", 0),
                    weight_alpha=float(tcfg.get("loss_weight_alpha", 0.3)),
                    attention_mask=batch.get("attention_mask"))
                # NaN/explosion guard: skip the update, don't poison the weights
                if not torch.isfinite(loss) or loss.item() > nan_threshold:
                    accel.print(f"[train] WARN non-finite/exploding loss "
                                f"({loss.item():.1f}) at step {step} — skipping update")
                    optim.zero_grad()
                    continue
                accel.backward(loss)
                if accel.sync_gradients:
                    accel.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()

            if accel.sync_gradients:
                step += 1
                tokens_seen += tokens_per_step
                if step % log_every == 0:
                    lv = accel.gather(loss.detach()).mean().item()
                    lr = sched.get_last_lr()[0]
                    now = time.time()
                    dt = max(now - t_last, 1e-6)
                    steps_per_sec = log_every / dt
                    tok_per_sec = steps_per_sec * global_batch * seq_len
                    t_last = now
                    accel.print(f"[train] epoch {epoch} step {step:5d} "
                                f"loss={lv:.4f} lr={lr:.2e} "
                                f"{tok_per_sec/1e3:.1f}k tok/s")
                    if accel.is_main_process:
                        gpu_gb = (torch.cuda.max_memory_allocated() / 1e9
                                  if torch.cuda.is_available() else 0.0)
                        with open(metrics_path, "a") as mf:
                            mf.write(json.dumps({
                                "t": now, "step": step, "epoch": epoch,
                                "loss": round(lv, 4), "lr": lr,
                                "tokens_seen": tokens_seen,
                                "tok_per_sec": round(tok_per_sec, 1),
                                "steps_per_sec": round(steps_per_sec, 3),
                                "gpu_gb": round(gpu_gb, 2),
                                "max_steps": max_steps or 0,
                            }) + "\n")
                    if tcfg.get("wandb"):
                        accel.log({"loss": lv, "lr": lr}, step=step)
                if step % save_every == 0:
                    _save(accel, model, tok, out_dir, step, train_step=step)
                if max_steps and step >= max_steps:
                    done = True
                    break

    _save(accel, model, tok, out_dir, step, final=True, train_step=step)
    if tcfg.get("wandb") and accel.is_main_process:
        accel.end_training()
    accel.print(f"[train] DONE — {step} steps, checkpoints in {out_dir}")


def _save(accel, model, tok, out_dir, step, final=False, train_step=0):
    accel.wait_for_everyone()
    tag = "final" if final else f"step{step}"
    path = os.path.join(out_dir, tag)
    unwrapped = accel.unwrap_model(model)
    if accel.is_main_process:
        os.makedirs(path, exist_ok=True)
        tok.save_pretrained(path)
        # save config.json too so the ckpt reloads with AutoModel (save_model omits it)
        unwrapped.config.save_pretrained(path)
    accel.save_model(unwrapped, path)
    # full training state (optim+sched+RNG) for --resume; + step counter
    resume_dir = os.path.join(out_dir, "resume")
    accel.save_state(resume_dir)
    if accel.is_main_process:
        with open(os.path.join(resume_dir, "berlin_step.txt"), "w") as f:
            f.write(str(train_step))
    accel.print(f"[train] saved checkpoint -> {path} (+resume state)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python train.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
