# Project Berlin on RunPod

RunPod has the powerful on-demand tier ThunderCompute lacks (H100/H200/B200/B300).

## GPU pick (EuroLLM-1.7B)
| GPU | VRAM | $/hr | Use |
|--|--|--|--|
| **H100 SXM5** | 80GB | ~$2.69 | best power/value — training + serving 1.7B |
| H200 SXM | 141GB | ~$3.59 | headroom for Pharia-7B |
| B200 SXM | 192GB | ~$4.99 | most powerful sane pick |
| B300 SXM6 | 288GB | ~$7.39 | overkill at 1.7B; only 70B+ |

VRAM is NOT the constraint at 1.7B (~22-34GB used). Pick on compute/throughput → H100 SXM5.

## Pod bring-up (on-demand GPU Pod, template = PyTorch 2.x CUDA 12.x)
```bash
git clone https://github.com/amar3012005/project-berlin.git && cd project-berlin
pip install transformers accelerate datasets pyyaml safetensors huggingface_hub
huggingface-cli login                      # gated EuroLLM + private model repo

# fetch corpora (same as Thunder)
python src/data/fetch_corpus.py --dataset wikimedia/wikipedia --config 20231101.de \
    --out data/wikipedia_de.jsonl --limit 40000 --min_chars 300
python src/data/fetch_corpus.py --dataset openlegaldata/court-decisions-germany \
    --config dump-20260520 --out data/openlegaldata_de.jsonl --limit 40000 --min_chars 200

# train (single H100 — no FSDP/DDP needed, 1.7B fits one GPU)
python src/diffusion/train.py configs/eurollm_cluster.yaml
# multi-GPU pod: accelerate launch --config_file configs/accelerate_ddp.yaml ...
```

## Resume a saved checkpoint (from HF Hub)
```bash
# after scripts/push_to_hub.py uploaded it:
python src/diffusion/infer.py --base <user>/project-berlin-eurollm-de \
    --ckpt <user>/project-berlin-eurollm-de --prompt "Das Gericht entschied" \
    --gen_len 40 --steps 16 --show_steps
```

## Serverless (inference endpoint)
RunPod Serverless: wrap `infer.py:generate` in a handler, deploy as endpoint, scales to
zero. Use for the diffusion-reveal API. See https://docs.runpod.io/serverless/overview
