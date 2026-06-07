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


class JsonlTextDataset(Dataset):
    def __init__(self, paths, text_field, tokenizer, max_length):
        self.tok = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.lines: list[str] = []
        for p in paths:
            fp = Path(p)
            if not fp.exists():
                raise FileNotFoundError(f"corpus file not found: {fp}")
            with fp.open(encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    self.lines.append(json.loads(ln)[text_field])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        enc = self.tok(
            self.lines[i],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0]}


def _collate(batch, pad_id):
    maxlen = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = b["attention_mask"]
    return {"input_ids": ids, "attention_mask": attn}


def build_dataloader(cfg_data, tokenizer, batch_size, shuffle=True):
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

        return DataLoader(ds, batch_size=batch_size, collate_fn=_collate_hf)

    ds = JsonlTextDataset(cfg_data["jsonl_paths"], cfg_data["text_field"],
                          tokenizer, cfg_data["max_length"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: _collate(b, pad_id))
