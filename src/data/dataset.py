"""Data loading for Project Berlin — local JSONL or HuggingFace datasets.

Returns a torch DataLoader yielding padded, tokenized batches. Same code path for
the toy 8-line sample and streamed multi-GB German corpora (OpenLegalData-DE,
DIN/VDE scrapes) — controlled entirely by the YAML `data:` block.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def _read_lines(paths, text_field, weights=None):
    """Read jsonl docs; optional per-file sampling weight (register balancing).
    weights: list[float] same length as paths; repeats files ~proportionally."""
    per_file: list[list[str]] = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"corpus file not found: {fp}")
        docs = []
        with fp.open(encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                docs.append(json.loads(ln)[text_field])
        per_file.append(docs)
    if not weights:
        return [d for docs in per_file for d in docs]
    # balance: scale each file's contribution by its weight (oversample by repeat)
    out = []
    maxw = max(weights)
    for docs, w in zip(per_file, weights):
        reps = max(1, round(w / maxw * (1 if len(docs) else 0)) or 1)
        out.extend(docs * reps)
    return out


class JsonlTextDataset(Dataset):
    """One doc per item, truncated/padded to max_length (variable length -> PAD)."""
    def __init__(self, paths, text_field, tokenizer, max_length, weights=None):
        self.tok = tokenizer
        self.max_length = max_length
        self.lines = _read_lines(paths, text_field, weights)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        enc = self.tok(self.lines[i], truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0]}


class PackedDataset(Dataset):
    """Sequence PACKING: concatenate all docs (eos-separated) and slice into fixed
    `block` chunks. Zero PAD waste, every token trains, STATIC shapes -> unlocks
    torch.compile + kernel fusion (the recompile-trap fix). Standard in LLaMA/LLaDA.

    FAST: per-doc truncation cap (avoid pathological huge legal docs) + BATCH
    tokenization (HF fast tokenizer parallelism) instead of one-doc-at-a-time. The
    naive untruncated single-doc loop stalled 20min on 65k docs — this is seconds."""
    def __init__(self, paths, text_field, tokenizer, block, weights=None,
                 doc_cap_blocks=8, batch_size=1000):
        self.block = block
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer.pad_token_id or 0
        cap = block * doc_cap_blocks                       # cap any single doc
        docs = _read_lines(paths, text_field, weights)
        stream: list[int] = []
        for i in range(0, len(docs), batch_size):
            enc = tokenizer(docs[i:i + batch_size], truncation=True,
                            max_length=cap, add_special_tokens=True)["input_ids"]
            for ids in enc:
                stream.extend(ids)
                stream.append(eos)
        n = (len(stream) // block) * block
        self.data = torch.tensor(stream[:n], dtype=torch.long).view(-1, block)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        ids = self.data[i]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _collate(batch, pad_id):
    maxlen = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = b["attention_mask"]
    return {"input_ids": ids, "attention_mask": attn}


def build_dataloader(cfg_data, tokenizer, batch_size, shuffle=True, num_workers=0):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if cfg_data.get("hf_dataset"):
        from datasets import load_dataset
        ds = load_dataset(
            cfg_data["hf_dataset"],
            cfg_data.get("hf_config"),
            split="train",
            streaming=cfg_data.get("streaming", False),
        )
        field = cfg_data["text_field"]
        maxlen = cfg_data["max_length"]

        def _tok(ex):
            enc = tokenizer(ex[field], truncation=True, max_length=maxlen)
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}

        ds = ds.map(_tok, remove_columns=ds.column_names if hasattr(ds, "column_names") else None)

        def _collate_hf(batch):
            maxl = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), maxl), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxl), dtype=torch.long)
            for i, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[i, :n] = torch.tensor(b["input_ids"])
                attn[i, :n] = torch.tensor(b["attention_mask"])
            return {"input_ids": ids, "attention_mask": attn}

        return DataLoader(ds, batch_size=batch_size, collate_fn=_collate_hf,
                          num_workers=num_workers)

    weights = cfg_data.get("corpus_weights")           # register balancing (optional)
    if cfg_data.get("pack"):                            # PACKED: static shapes, no PAD
        ds = PackedDataset(cfg_data["jsonl_paths"], cfg_data["text_field"],
                           tokenizer, cfg_data["max_length"], weights=weights)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, drop_last=True)

    ds = JsonlTextDataset(cfg_data["jsonl_paths"], cfg_data["text_field"],
                          tokenizer, cfg_data["max_length"], weights=weights)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: _collate(b, pad_id),
                      num_workers=num_workers,
                      pin_memory=True,
                      persistent_workers=num_workers > 0)
