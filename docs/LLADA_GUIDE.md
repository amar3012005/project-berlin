# What you can do with LLaDA-8B (and how)

A practical playbook for leveraging the fully-trained open **diffusion language model**
**LLaDA-8B** from your AR→diffusion pipeline in this repo.

LLaDA ("Large Language Diffusion with mAsking") is an 8B **masked diffusion** LM trained
from scratch on **2.3 trillion tokens** (then SFT'd on **4.5M prompt–response pairs**). It
is *not* autoregressive: it generates by starting from an all-`[MASK]` answer region and
**iteratively denoising** (predict-all-tokens-at-once, commit the confident ones, remask
the rest) over a fixed number of steps. On standard benchmarks it is competitive with
LLaMA3-8B (e.g. MMLU 65.9 5-shot, GSM8K 70.7 4-shot for the Base model).

Primary sources (cite these, all claims below trace to them):

| Ref | What | Link |
|---|---|---|
| **arXiv:2502.09992** | LLaDA — Large Language Diffusion Models | https://arxiv.org/abs/2502.09992 |
| **arXiv:2505.19223** | LLaDA 1.5 — VRPO preference alignment | https://arxiv.org/abs/2505.19223 |
| **arXiv:2504.12216** | d1 / diffu-GRPO — RL reasoning for dLLMs | https://arxiv.org/abs/2504.12216 |
| repo | `ML-GSAI/LLaDA` (code, GUIDELINES.md, EVAL.md) | https://github.com/ML-GSAI/LLaDA |

HF weights: `GSAI-ML/LLaDA-8B-Instruct`, `GSAI-ML/LLaDA-8B-Base`, and the 1.5 alignment
checkpoint `GSAI-ML/LLaDA-1.5`. The reference `generate()` / `get_log_likelihood()` code
ships in the repo (`generate.py`, `get_log_likelihood.py`); this repo already wires the
model into `serve/app.py --mode llada` and mirrors the same forward-masking / denoise math
in `src/diffusion/diffusion.py` and `src/diffusion/generate.py`.

> **Where this fits your pipeline.** Your `src/diffusion/*` trains a *small* AR base
> (Qwen/EuroLLM) into a diffusion LM as a learning harness. LLaDA-8B is the *production-grade*
> version of exactly that idea — already trained. The sections below are ordered cheapest →
> most expensive: run it, adapt it (LoRA-SFT), re-pretrain it, RL it, eval it, serve it.

> **GPU shorthand** (used throughout): **A6000** = 48 GB VRAM (Ampere, ~$0.40–0.80/hr
> rented); **H100** = 80 GB (Hopper, ~$2–4/hr rented). LLaDA-8B in bf16 weights ≈ 16 GB;
> with activations/KV-free diffusion buffers inference sits **~16–20 GB**, so a single
> A6000 runs it comfortably. Dollar figures below assume rented cloud GPUs at those rates.

---

## 1. RUN AS-IS — load `LLaDA-8B-Instruct` and generate

**VRAM:** ~16–20 GB bf16 → fits one **A6000 (48 GB)** with huge headroom; trivial on H100.
**Cost/time:** generation only — pennies. One 256-token answer at `steps=256` takes a few
seconds on H100, ~10–20 s on A6000.

```bash
pip install "transformers==4.38.2" accelerate torch   # repo-pinned version for trust_remote_code
```

```python
# run_llada.py — minimal, mirrors ML-GSAI/LLaDA generate.py
import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate   # vendored from github.com/ML-GSAI/LLaDA

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda").eval()

prompt = "Erkläre in zwei Sätzen, warum der Himmel blau ist."
msg = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
input_ids = tok(text, return_tensors="pt")["input_ids"].to("cuda")

out = generate(
    model, input_ids,
    steps=128,                  # # of denoising rounds (more = better, slower)
    gen_length=128,             # length of the all-[MASK] answer region
    block_length=32,            # semi-autoregressive block size (gen_length % block == 0)
    temperature=0.0,            # 0 = greedy/argmax commit
    cfg_scale=0.0,              # unsupervised classifier-free guidance (0 = off)
    remasking="low_confidence", # 'low_confidence' (recommended) or 'random'
    mask_id=126336,             # LLaDA's [MASK] token id
)
print(tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
```

**The `generate()` knobs (exact signature from the repo):**

| Param | Default | Meaning / how to tune |
|---|---|---|
| `steps` | 128 | Denoising iterations. Quality scales with steps; set `steps == gen_length` for ~1 token/step. Fewer steps = faster but lower quality. |
| `gen_length` | 128 | Size of the fully-masked answer block. This is the **max answer length**, fixed up front (no EOS-driven early stop like AR). |
| `block_length` | 128 | Semi-autoregressive remasking: split `gen_length` into left-to-right blocks of this size; finish a block before moving on. `gen_length` must be divisible by it. Smaller blocks (e.g. 32) often improve coherence. |
| `temperature` | 0.0 | Sampling temp on the categorical commit. 0 = argmax. |
| `cfg_scale` | 0.0 | Classifier-free guidance strength. >0 (e.g. 1.0–2.0) sharpens prompt-following at ~2× compute (runs a masked-prompt forward pass too). |
| `remasking` | `'low_confidence'` | After predicting all masked tokens, **keep the most-confident** predictions and **remask the rest** for the next round. `'low_confidence'` (recommended) vs `'random'`. |
| `mask_id` | 126336 | The `[MASK]` token id baked into LLaDA's tokenizer. |

**How the "diffusion effect" appears.** Each step the bidirectional Transformer predicts
*every* masked position at once; `low_confidence` remasking commits the top
`s/t` fraction and remasks the rest. So the answer materializes as a field of `▒` masks
that fill in **non-left-to-right** — the "Inception/denoising reveal." This repo renders
exactly that animation: `serve/app.py` streams the partial sequence at each denoising step
over SSE, rendering still-masked positions as `▒` (see §6). The same reveal logic lives in
`src/diffusion/generate.py::generate(show_steps=True)`.

**Quick consistency check** (per the repo's EVAL guidance): generation quality is stable
across `gen_length/steps/block_length ∈ {(256,256,256),(512,512,512),(1024,1024,1024)}` —
pick the smallest that fits your latency budget.

---

## 2. LoRA-SFT for German / a domain

**Goal:** teach LLaDA-8B German register or domain style/format **cheaply**, without the
4.5M-pair full SFT. PEFT-LoRA on the attention/MLP projections needs **far fewer pairs**
(a few thousand to ~50k good pairs gets you a strong domain adapter).

**The masked-diffusion SFT objective (this is the one rule that matters).** SFT is *not*
next-token CE. You:

1. Concatenate `prompt + response` (chat-templated), pad the response with `<EOS>` to a
   fixed length.
2. **Mask ONLY the response tokens** — the prompt stays clean and fully visible. (This is
   the sole difference from pretraining, which masks everything.)
3. Forward-pass the noisy sequence; compute **cross-entropy only on masked response
   positions**, scaled by the per-example mask probability and answer length.

Reference objective (from LLaDA `GUIDELINES.md`, matches `src/diffusion/diffusion.py`):

```python
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)        # random mask ratio per example
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy = torch.where(masked, 126336, input_ids)     # 126336 = [MASK]
    return noisy, masked, p_mask

# --- SFT step ---
noisy, masked, p_mask = forward_process(input_ids)
noisy[prompt_mask] = input_ids[prompt_mask]            # KEEP PROMPT CLEAN  <-- the key line
masked = masked & response_mask                        # only score response positions
logits = model(noisy).logits
token_loss = F.cross_entropy(logits[masked], input_ids[masked], reduction="none") / p_mask[masked]
loss = (token_loss / answer_lengths[masked]).sum() / input_ids.shape[0]
```

**Wrap with PEFT-LoRA** (note: not yet vendored in this repo — `pip install peft`, then):

```python
from peft import LoraConfig, get_peft_model
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)   # trains ~0.1-0.3% of params
# then run the masked-diffusion SFT loop above; AdamW lr ~1e-4 to 2e-4
```

**Dataset format** — JSONL of chat pairs (one per line), same shape your
`src/data/dataset.py` packer (`PackedDataset` / `build_dataloader`) already expects for
`{"text": ...}`, but with roles:

```json
{"prompt": "Fasse diesen Absatz zusammen: ...", "response": "Zusammenfassung: ..."}
```

Apply `tok.apply_chat_template` to build `prompt`, keep `prompt_lengths` so you can build
`prompt_mask`/`response_mask`. Pad each `response` with `<EOS>` to the batch's fixed length;
**1% of examples should use a random length in U[1, 4096]** so the model handles variable
answer lengths (LLaDA's exact recipe).

**How many pairs / time / cost (LoRA, 1× GPU):**

| Pairs | GPU | Wall time | ~$ |
|---|---|---|---|
| 3k–10k (style/format adapter) | A6000 48GB | 1–3 h | $1–3 |
| 20k–50k (domain German) | H100 80GB | 3–8 h | $10–30 |
| Full SFT (4.5M, full FT) | 8×H100 | days | $1k+ (rarely needed; use LoRA) |

LoRA fits in <24 GB at bf16 + r=16, so even the A6000 handles SFT. Save just the adapter
(~50–200 MB); load it on top of the base for serving.

---

## 3. CONTINUED PRETRAIN on German

**When to do this instead of SFT:** SFT teaches *behavior on tasks you have pairs for*.
Continued pretrain (a.k.a. domain-adaptive pretraining) teaches **raw language/knowledge**
when (a) LLaDA's German is weak at the *token/fluency* level, or (b) you have lots of
**unlabeled** German text but few instruction pairs. Rule of thumb: if generations are
*fluent but off-task* → SFT; if they're *disfluent / wrong vocabulary / code-switching* →
continued pretrain first, then a small SFT.

**Objective:** identical masked-denoising as pretraining, but now you **mask everything**
(no clean-prompt carve-out) over streamed German tokens. This is exactly what
`src/diffusion/diffusion.py::denoise_loss` + `forward_mask` already implement, and what
`src/diffusion/train.py` drives. You can point that trainer's data config at German.

```bash
# 1) build a German corpus (already wired in this repo)
bash scripts/fetch_german.sh            # default: wikimedia/wikipedia 20231101.de (ungated)
# for scale, stream FineWeb-2 deu_Latn directly (see configs/german_scaled.yaml):
#   data.hf_dataset: HuggingFaceFW/fineweb-2 ; data.hf_config: deu_Latn ; data.streaming: true

# 2) continued pretrain (full or LoRA). 8-GPU example mirrors src/diffusion/train.py header:
accelerate launch --config_file configs/accelerate_8xh100.yaml \
    src/diffusion/train.py configs/german_scaled.yaml
```

For LLaDA-8B specifically, prefer **LoRA continued-pretrain** unless you have a real cluster:
full-FT of 8B needs FSDP across 8×H100/A100 (`configs/accelerate_fsdp.yaml`).

**Token budget / cost (rough):**

| Goal | Tokens | Hardware | Time | ~$ |
|---|---|---|---|---|
| Light German nudge (LoRA) | 0.5–2 B | 1×H100 | 12–48 h | $30–150 |
| Solid German register | 10–30 B | 8×H100 (FSDP) | 1–4 days | $1k–5k |
| (Reference: LLaDA from scratch) | 2.3 T | — | 0.13M H800-hr | — |

Always finish a continued-pretrain run with a **small LoRA-SFT (§2)** to re-instill
instruction-following.

---

## 4. REASONING — diffu-GRPO / d1 (RL for diffusion LMs)

To push LLaDA-8B on math/logic (GSM8K, MATH500, Countdown, Sudoku), use **d1** — a two-stage
recipe (**arXiv:2504.12216**): (a) masked-SFT on reasoning traces, then (b) **diffu-GRPO**,
the first policy-gradient RL for masked dLLMs (critic-free, GRPO-style).

**What it needs:**
- **Base:** `LLaDA-8B-Instruct` (the d1 paper instantiates on exactly this).
- **SFT data:** `s1k` — 1,000 curated high-quality reasoning traces (cheap masked-SFT, §2).
- **A rule-based reward/verifier** (no learned reward model): a **composed reward = format
  reward + correctness reward**, where correctness is exact-match of the extracted final
  answer against ground truth. One trained model **per task**.
- **Rollouts:** online generations during RL, capped to **gen length 256** to control cost.
- **The efficiency trick:** diffu-GRPO estimates per-token log-probs in **one step** via a
  **mean-field approximation with random prompt masking** — random masking beats fixed
  masking and lets you scale `μ` (gradient updates per batch) far higher, so you get many
  policy updates per expensive rollout batch.

**Expected gains (d1 Table 1, 0-shot accuracy; baseline → d1-LLaDA), eval at gen len 256:**

| Task | LLaDA-8B-Instruct | +SFT | +diffu-GRPO | **d1-LLaDA (SFT+RL)** |
|---|---|---|---|---|
| GSM8K | 76.7 | 78.8 | 79.8 | **81.1** |
| MATH500 | 32.4 | 32.6 | 37.2 | **38.6** |
| Countdown | 19.5 | 14.5 | 31.3 | **32.0** |
| Sudoku | 6.7 | 8.5 | 12.9 | **16.7** |

Headline deltas (d1 reports best-setting gains): **GSM8K +3.9%, MATH500 +4.0%, Countdown
+26.2%, Sudoku +10.0%.** Math gains are smaller because LLaDA is already near-saturated
there; logic/planning tasks have the most headroom. `diffu-GRPO` alone beats `SFT` alone in
all 12 setups; combining them (d1) wins in 11/12.

**Run it** (from `github.com/dllm-reasoning/d1`):

```bash
# Stage A: masked-SFT on s1k (2 GPUs)
cd SFT && CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ddp_config.yaml \
    --main_process_port 29500 --num_processes 2 \
    sft_train.py --grad_accum_steps 4 --batch_size 1 --num_epochs 20

# Stage B: diffu-GRPO RL on the SFT checkpoint (8 GPUs)
cd ../diffu-GRPO && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
```

**Cost/time:** SFT-on-s1k is hours on 2×H100 (~$10–30). diffu-GRPO RL is the expensive part
— online rollouts on **8 GPUs**; budget **1–3 days on 8×H100 (~$400–2k)** per task. On
limited hardware, run SFT + a *short* diffu-GRPO and stop early — most of the lift on
logic tasks appears within the first chunk of RL steps.

> Adjacent alignment option: **LLaDA 1.5 (arXiv:2505.19223)** applies **VRPO**
> (Variance-Reduced Preference Optimization — DPO for masked diffusion, with antithetic
> sampling + optimal MC budget to cut ELBO-gradient variance). Gains over SFT-only LLaDA:
> **GSM8K +4.7, HumanEval +3.0, MBPP +1.8, IFEval +4.0, Arena-Hard +4.3.** Use VRPO for
> *general alignment/helpfulness*; use d1 for *task-specific reasoning*.

---

## 5. EVAL — `lm-evaluation-harness` with LLaDA's conditional paths

LLaDA cannot be scored like an AR model. The repo's `EVAL.md` provides an
`lm-evaluation-harness` integration with **two evaluation paths**:

**A) Conditional-likelihood (PPL) tasks** — score candidates by LLaDA's masked-diffusion
log-likelihood (Monte-Carlo over mask ratios), pick argmax. Report **accuracy**:
- LAMBADA, HellaSwag, MMLU, CMMLU, C-Eval, ARC-C, PIQA, WinoGrande, TruthfulQA, GPQA.

**B) Conditional-generation tasks** — actually run `generate()` and check the answer.
Report **exact-match / pass@1**:
- GSM8K, Minerva MATH, BBH, HumanEval, HumanEval-FIM, MBPP.

```bash
# install the harness + LLaDA's eval glue (see EVAL.md / eval_llada_lm_eval.sh)
pip install lm-eval

# (A) likelihood task — no CFG, conditional likelihood path
accelerate launch eval_llada.py --tasks mmlu,hellaswag,arc_challenge \
    --confidence cmf --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False

# (B) generation task — set matched gen_length/steps/block_length
accelerate launch eval_llada.py --tasks gsm8k,humaneval \
    --model llada_dist --gen_length 256 --steps 256 --block_length 256 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',cfg=0.0
```

**What to report.** Match the LLaDA paper's setup so numbers are comparable: MMLU 5-shot,
GSM8K 4-shot, HumanEval 0-shot, MATH 4-shot. Reference Base-model targets to sanity-check
your harness: **MMLU 65.9, GSM8K 70.7, HumanEval 33.5, MATH 27.3.** Generation results are
stable across `(gen_length,steps,block_length)` of 256/512/1024 — report the setting you
used. For your own German adapters (§2/§3), reuse `src/diffusion/eval.py` for masked
denoise-perplexity on a held-out German `eval.jsonl`.

---

## 6. SERVE — chat UI with the diffusion-reveal effect

This repo already ships the demo. `serve/app.py` loads **one** backend at startup and
streams the reverse-diffusion denoising over SSE, rendering still-masked positions as `▒`
so the UI animates `▒ → word` non-left-to-right.

```bash
# serve the real LLaDA-8B-Instruct (trust_remote_code, bf16, cuda) — fits one A6000
python serve/app.py --mode llada --port 8890

# (contrast) serve your own AR->diffusion checkpoint:
python serve/app.py --mode berlin --ckpt checkpoints/qwen_long/step7400 \
    --base Qwen/Qwen2.5-0.5B --port 8890
```

Endpoints: `GET /api/health` → `{"ok", "mode", "model"}`; the SSE chat stream emits the
**full current generation each denoising step** with masks as `U+2592` (`▒`). Generation is
serialized behind a global lock (single-GPU, one request at a time), model loaded once.
**VRAM:** `--mode llada` ≈ 16–20 GB bf16 → one A6000 is enough; an H100 gives snappier
multi-step reveals. To serve a LoRA adapter from §2, load the base then
`PeftModel.from_pretrained(model, adapter_dir)` before the generation loop.

---

## Decision table — want X → do Y, ~$Z, ~T hours

| You want… | Do (section) | Hardware | ~Time | ~$ |
|---|---|---|---|---|
| Just run LLaDA-8B / demo it | Run as-is (§1) + serve (§6) | 1×A6000 | minutes | <$1 |
| German/domain *style & format* | LoRA-SFT, 3–10k pairs (§2) | 1×A6000 | 1–3 h | $1–3 |
| Strong German *task* behavior | LoRA-SFT, 20–50k pairs (§2) | 1×H100 | 3–8 h | $10–30 |
| Fix *fluency/vocabulary* in German | Continued pretrain LoRA (§3) + small SFT | 1×H100 | 12–48 h | $30–150 |
| Native-grade German register | Continued pretrain 10–30B tok (§3) + SFT | 8×H100 FSDP | 1–4 d | $1k–5k |
| Better math/logic reasoning | d1: SFT(s1k) + diffu-GRPO (§4) | 2→8×H100 | 1–3 d/task | $400–2k |
| Better general alignment/helpfulness | VRPO / LLaDA 1.5 (§4 note) | 4–8×H100 | 1–2 d | $300–1.5k |
| Benchmark numbers I can trust | lm-eval harness, both paths (§5) | 1×H100 | 1–6 h | $5–30 |
| Live chat with diffusion reveal | `serve/app.py --mode llada` (§6) | 1×A6000 | minutes | <$1 |

---

### Notes & caveats
- **Pin `transformers==4.38.2`** for LLaDA's `trust_remote_code` path (repo requirement);
  this repo's own `requirements.txt` pins a newer transformers for the *small-model* trainer
  — keep LLaDA in a separate env, or vendor `generate.py`/`get_log_likelihood.py` from
  `ML-GSAI/LLaDA` directly.
- **No EOS early-stop:** diffusion fixes answer length at `gen_length` up front — size it to
  your longest expected answer; over-long wastes compute, too-short truncates.
- **Cost figures are rough** rented-GPU estimates (A6000 ~$0.4–0.8/hr, H100 ~$2–4/hr) and
  scale with steps × gen_length × rollouts; treat them as order-of-magnitude.
- **Always re-instill instructions** with a small LoRA-SFT after any continued-pretrain.
