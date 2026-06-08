#!/usr/bin/env bash
# =============================================================================
# fetch_german.sh — build data/german_corpus.jsonl for the REAL German run
# (configs/german_scaled.yaml). Streams a large GERMAN corpus from HuggingFace
# via src/data/fetch_corpus.py (no full multi-TB download), filters by min length,
# caps doc count, and writes {"text": ...} jsonl that PackedDataset packs + caches.
#
# DEFAULT SOURCE — wikimedia/wikipedia 20231101.de:
#   Fully OPEN + UNGATED, clean encyclopedic German, no auth/license gate. Streams
#   reliably anywhere (CI, fresh pod) with zero token. ~500k docs * ~1-2k chars
#   gives a solid base German register for a first real 0.5B diffusion LM.
#
# PREFERRED HIGH-QUALITY OPTION — German FineWeb-2 (HuggingFaceFW/fineweb-2,
# config "deu_Latn"):
#   The best modern web-scale German corpus (deduped, quality-filtered, trillions
#   of tokens). It is the right source to actually hit ~30B tokens. Either:
#     (a) stream it directly from the trainer — in german_scaled.yaml set:
#           data.hf_dataset: HuggingFaceFW/fineweb-2
#           data.hf_config:  deu_Latn
#           data.streaming:  true
#         (no local jsonl needed — dataset.py hf_dataset path streams it), OR
#     (b) materialize a jsonl slice here by switching DATASET/CONFIG below.
#   Left as a comment because it is large; Wikipedia is the safe ungated default.
#
# OTHER GERMAN SOURCES (uncomment to use):
#   * uonlp/CulturaX        config "de"   — huge, but GATED (needs HF auth + accept)
#   * oscar-corpus/OSCAR-2301 config "de" — huge, but GATED (needs HF auth + accept)
#   germanquad below is an optional small supplement (open, QA register).
#
# USAGE:
#   bash scripts/fetch_german.sh
#   LIMIT=1000000 bash scripts/fetch_german.sh        # override doc cap
# =============================================================================
set -euo pipefail

# Resolve repo root from this script's location (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${OUT:-data/german_corpus.jsonl}"
LIMIT="${LIMIT:-500000}"          # large fetch: ~500k docs
MIN_CHARS="${MIN_CHARS:-300}"     # drop stubs / boilerplate

# --- DEFAULT: open, ungated German Wikipedia --------------------------------
DATASET="${DATASET:-wikimedia/wikipedia}"
CONFIG="${CONFIG:-20231101.de}"

# --- PREFERRED high-quality (uncomment to materialize a FineWeb-2 slice) -----
# DATASET="HuggingFaceFW/fineweb-2"
# CONFIG="deu_Latn"

mkdir -p "$(dirname "${OUT}")"

echo "[fetch_german] dataset=${DATASET} config=${CONFIG} -> ${OUT}"
echo "[fetch_german] limit=${LIMIT} min_chars=${MIN_CHARS}"

python src/data/fetch_corpus.py \
  --dataset "${DATASET}" \
  --config "${CONFIG}" \
  --split train \
  --out "${OUT}" \
  --limit "${LIMIT}" \
  --min_chars "${MIN_CHARS}"

# --- OPTIONAL: append open German QA register (germanquad) -------------------
# Small, open, adds a different German register. Appends to the SAME jsonl so the
# packer concatenates everything. Uncomment to include:
# python src/data/fetch_corpus.py \
#   --dataset community-datasets/germanquad \
#   --out "${OUT}.germanquad" \
#   --text_field context \
#   --limit 50000 --min_chars 200
# cat "${OUT}.germanquad" >> "${OUT}" && rm -f "${OUT}.germanquad"

echo "[fetch_german] DONE -> ${OUT}"
wc -l "${OUT}" || true
