# Project Berlin — Sovereign German Diffusion LM

AR → discrete-diffusion "surgery" (ICLR 2025 / LLaDA / SEDD). Convert an
autoregressive German LM into a masked-diffusion LM — **no training from scratch**.

Architecture-agnostic: identical code runs on the **124M german-gpt2** (GPT2-arch,
laptop/cheap GPU) and **Aleph-Alpha/Pharia-1-LLM-7B-control** (llama-arch, 8×A100).
Swapping models is a one-line YAML change — both hit the same `create_causal_mask`
surgery point. **No rebuild on the GPU instance.**

## Layout
```
configs/   toy_gpt2.yaml · pharia_7b.yaml · accelerate_fsdp.yaml
src/diffusion/  load_model · attention_surgery · diffusion · generate · train · train_overfit
src/data/  dataset.py (jsonl + HF datasets, streaming)
data/      sample_de.jsonl   (replace with OpenLegalData-DE + DIN/VDE scrapes)
setup_env.sh   start.sh
```

## Proofs (laptop, $0)
```bash
./start.sh selftest          # M0-M4 all green
```
| Milestone | Stage | Proof |
|--|--|--|
| M0 load | base init | german-gpt2 loads, coherent German |
| M1 rewire | attention surgery | future token shifts pos0 logits 14.18 (impossible under AR) |
| M2 mask | LLaDA forward t~U[0,1] | CE on masked tokens, 1/t NELBO, backprops |
| M3 sample | reverse / FS-DFM | parallel unmask fills all positions |
| M4 overfit | data-constrained | loss 34.6→3.8, recon 0%→27% |

## Train
```bash
# laptop / single GPU
./start.sh train configs/toy_gpt2.yaml

# 8-GPU FSDP on ThunderCompute
./start.sh train_multi configs/pharia_7b.yaml
```

## GPU instance bring-up (ThunderCompute)
```bash
git clone <repo> && cd berlin
bash setup_env.sh                 # uv venv + CUDA torch + accelerate (+ optional flash-attn)
source .venv/bin/activate
huggingface-cli login             # Pharia-1 is gated
./start.sh train_multi configs/pharia_7b.yaml
```

## Roadmap
- [x] M0–M4 toy pipeline proven
- [x] Architecture-agnostic surgery + config-driven train + FSDP launch
- [ ] M5 real corpora (OpenLegalData-DE, DIN/VDE) → `data/*.jsonl`
- [ ] M6 Pharia-7B Stage-2 pre-train (8×A100, 50–100B tok, ~$1.5–3k)
- [ ] M7 DRAKES alignment (Sie/du, DIN hallucination penalty)
- [ ] M8 FS-DFM 4–8 step distill + Inception blur UI
