"""Fetch German corpora from HuggingFace -> data/*.jsonl ({"text": ...}).

Streams (no full download), extracts a text field, filters by min length, caps
doc count. Used to build the real Stage-2 training corpora:

  # German court rulings (OpenLegalData, public-domain under German law)
  python src/data/fetch_corpus.py --dataset openlegaldata/court-decisions-germany \
      --out data/openlegaldata_de.jsonl --limit 50000 --min_chars 200

  # German technical/encyclopedic register (clean proxy for copyrighted DIN/VDE)
  python src/data/fetch_corpus.py --dataset wikimedia/wikipedia --config 20231101.de \
      --out data/wikipedia_de.jsonl --limit 50000 --min_chars 300

DIN/VDE standards themselves are copyrighted and cannot be redistributed — do not
scrape them. Court decisions + Wikipedia give legal + technical German register.
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

# candidate text fields tried in order when --text_field not given
TEXT_FIELDS = ["text", "content", "decision_text", "body",
               "entscheidungsgruende", "tatbestand", "tenor", "article"]


def pick_text(example: dict, explicit: str | None) -> str | None:
    if explicit:
        v = example.get(explicit)
        return v if isinstance(v, str) and v.strip() else None
    parts = []
    for f in TEXT_FIELDS:
        v = example.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n".join(parts) if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    print(f"[fetch] {args.dataset} config={args.config} split={args.split} "
          f"-> {args.out} (limit={args.limit}, min_chars={args.min_chars})")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = seen = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in ds:
            seen += 1
            txt = pick_text(ex, args.text_field)
            if not txt or len(txt) < args.min_chars:
                continue
            fh.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 5000 == 0:
                print(f"[fetch] kept {kept}/{args.limit} (scanned {seen})")
            if kept >= args.limit:
                break
    print(f"[fetch] DONE: wrote {kept} docs to {args.out} (scanned {seen})")


if __name__ == "__main__":
    main()
