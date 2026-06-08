"""Overfit check: deterministic masked-CE on SEEN(train_sub) vs UNSEEN(val).
Same seed/mask/data order across checkpoints -> directly comparable.
Healthy: val-CE drops with steps and tracks train-CE.
Overfit: val plateaus/rises while train keeps dropping (gap widens)."""
import sys, glob, json, argparse
sys.path.insert(0, "/workspace/project-berlin/src/diffusion")
sys.path.insert(0, "/workspace/project-berlin/src/data")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from attention_surgery import apply_blockwise_bidirectional
from diffusion import MASK_TOKEN
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--ckpts", required=True)            # comma list of step dirs
ap.add_argument("--block", type=int, default=512)
ap.add_argument("--n_blocks", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--mask_rate", type=float, default=0.5)
a = ap.parse_args()
dev = "cuda"
ckpts = a.ckpts.split(",")
tok = AutoTokenizer.from_pretrained(ckpts[0])
mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
eos = tok.eos_token_id or 0


def pack(path):
    ids = []
    for ln in open(path):
        t = json.loads(ln)["text"]
        ids += tok(t, truncation=True, max_length=a.block * 4,
                   add_special_tokens=True)["input_ids"] + [eos]
    n = (len(ids) // a.block) * a.block
    return torch.tensor(ids[:n]).view(-1, a.block)[:a.n_blocks]


datasets = {
    "train_seen": pack("/workspace/project-berlin/data/train_sub.jsonl"),
    "val_unseen": pack("/workspace/project-berlin/data/val.jsonl"),
}
for k, v in datasets.items():
    print(f"[data] {k}: {tuple(v.shape)} blocks")

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
model.resize_token_embeddings(len(tok), mean_resizing=False)
model.to(dev).eval()
apply_blockwise_bidirectional(model, block_size=64)


@torch.no_grad()
def eval_ce(blocks):
    tot = 0.0
    cnt = 0
    for bi in range(0, blocks.size(0), a.batch):
        torch.manual_seed(1234 + bi)                 # identical masks across ckpts
        x = blocks[bi:bi + a.batch].to(dev)
        m = torch.rand(x.shape, device=dev) < a.mask_rate
        m[:, 0] = True
        noised = torch.where(m, mask_id, x)
        logits = model(input_ids=noised).logits
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             x.view(-1), reduction="none").view_as(x)
        tot += (ce * m).sum().item()
        cnt += int(m.sum().item())
    return tot / max(cnt, 1)


head = "checkpoint".rjust(10) + " | " + "train_seen_CE".rjust(13) + " | " \
       + "val_unseen_CE".rjust(13) + " | " + "gap".rjust(7)
print("\n" + head)
print("-" * len(head))
for ck in ckpts:
    st = {}
    for s in sorted(glob.glob(ck + "/*.safetensors")):
        st.update(load_file(s))
    st = {kk.replace("_orig_mod.", ""): vv for kk, vv in st.items()}
    model.load_state_dict(st, strict=False)
    tr = eval_ce(datasets["train_seen"])
    va = eval_ce(datasets["val_unseen"])
    name = ck.rstrip("/").split("/")[-1]
    print(name.rjust(10) + " | " + f"{tr:13.4f}" + " | " + f"{va:13.4f}"
          + " | " + f"{va - tr:7.4f}", flush=True)
