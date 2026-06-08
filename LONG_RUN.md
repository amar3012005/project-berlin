# Project Berlin — Long Run Playbook (crash-safe, resumable, cap-respecting)

Goal: train to **real quality** (≥1B tokens) across multiple ≤3-4hr sessions, surviving
crashes AND pod deletion. Every piece below is already debugged this session.

## Why a NETWORK VOLUME (the one missing piece)
A pod's own disk is deleted with the pod. A long run spans many capped sessions →
checkpoints MUST outlive the pod. RunPod **network volumes** persist independently and
re-attach to any new pod. Checkpoints + HF cache live there → resume across sessions.

## Step 1 — one-time: create network volume (~150GB)
```
create-network-volume name=berlin-vol size=150 dataCenterId=US-TX-6
```
(US-TX-6 = where B200 lives. ~$0.07/GB/mo ≈ $10/mo. Delete when project done.)

## Step 2 — each session: create pod attached to the volume
```
create-pod name=berlin-b200 gpuTypeIds=["NVIDIA B200"] gpuCount=1 cloudType=SECURE
  imageName=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
  volumeMountPath=/workspace  networkVolumeId=<berlin-vol id>  ports=["8888/http","22/tcp"]
  env={PUBLIC_KEY: <your ssh pubkey>}
```

## Step 3 — bootstrap (deps wiped each pod; volume persists)
```bash
cd /workspace && [ -d project-berlin ] || git clone https://github.com/amar3012005/project-berlin.git
cd project-berlin && git pull
pip install -q --break-system-packages transformers accelerate datasets pyyaml safetensors huggingface_hub
huggingface-cli login --token <HF_TOKEN>
# corpora persist on the volume; fetch only if missing:
[ -f data/wikipedia_de.jsonl ] || python src/data/fetch_corpus.py --dataset wikimedia/wikipedia --config 20231101.de --out data/wikipedia_de.jsonl --limit 40000 --min_chars 300
[ -f data/openlegaldata_de.jsonl ] || python src/data/fetch_corpus.py --dataset openlegaldata/court-decisions-germany --config dump-20260520 --out data/openlegaldata_de.jsonl --limit 40000 --min_chars 200
```

## Step 4 — launch (self-healing, auto-resume)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src/diffusion:src/data PYTHONUNBUFFERED=1
# go.sh: kills stragglers, resumes from checkpoints/eurollm_cluster/resume if present, loops on crash
cd /workspace && setsid bash go.sh > /tmp/go.log 2>&1 < /dev/null &
# monitor: https://<podid>-8888.proxy.runpod.net
```
Uses `configs/eurollm_long.yaml`: **save_every 100** (crash-safe), max_steps 200k,
batch 32 / seq 512, no-grad-ckpt + compile (~77k tok/s burst on B200).

## Crash-safety (the bit you asked for)
- **save_every: 100** → checkpoint + full resume-state (model+optim+sched+RNG) every 100 steps.
- **go.sh auto-resume loop** → on crash, relaunches from `checkpoints/eurollm_cluster/resume`.
- **OOM-resilient step** → transient OOM skips a batch, no hard crash / zombie.
- **compile-prefix fixed** → checkpoints reload correctly (the silent base-model bug, fixed 5897b4c).
- Network volume → all of the above survives pod deletion across sessions.

## Cap discipline (USER RULE: never >3-4 hrs/session)
- Each session: launch → train (checkpoints every 100) → **STOP pod before 3-4hr mark**.
- Next session: recreate pod on same volume → go.sh auto-resumes → continue.
- Stack sessions until ≥1B tokens seen (watch `tokens_seen` in metrics, not `step`).

## Sanity gate before trusting the model
Run `src/diffusion/eval.py` — pseudo-perplexity should drop from ~1020 toward tens as
tokens_seen climbs into the billions. If it stays ~1000, something's wrong (not just slow).
