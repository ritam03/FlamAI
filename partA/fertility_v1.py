#!/usr/bin/env python3
"""
fertility_v1.py — Corrected tokenizer fertility benchmark (v1)

Fixes from v0 (fertility.py):
  1. BUG FIX: word splitting uses str.split() instead of str.split(" ")
     to avoid empty tokens from double-spaces in the corpus.
  2. CONCEPTUAL FIX: adds per-sentence token count (the metric that
     actually matters for cost comparison on parallel corpora).
  3. Adds per-grapheme-cluster and per-UTF8-byte denominators for
     comprehensive cross-language comparison.
  4. Supports multiple tokenizers in a single run for side-by-side
     comparison.

Usage:
    python fertility_v1.py --corpus eng=corpus/eng.txt \
                           --corpus hin=corpus/hin.txt \
                           --corpus kan=corpus/kan.txt \
                           --corpus tam=corpus/tam.txt \
                           --tokenizer gpt2 \
                           --tokenizer cl100k_base \
                           --tokenizer hf:google/gemma-2-2b \
                           --output results/fertility_results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata

# ─── grapheme cluster segmentation ───────────────────────────────────────────
# We use the regex module for \X (extended grapheme cluster), falling back
# to a character-level count if regex is unavailable.
try:
    import regex
    def count_grapheme_clusters(text: str) -> int:
        """Count Unicode extended grapheme clusters (UAX #29)."""
        return len(regex.findall(r'\X', text))
except ImportError:
    def count_grapheme_clusters(text: str) -> int:
        """Fallback: count characters (approximate grapheme count)."""
        return len(text)


def load_tokenizer(spec: str):
    """Load a tokenizer by spec string.
    
    Specs:
        gpt2, cl100k_base, o200k_base  -> tiktoken encodings
        hf:<repo_id>                    -> HuggingFace AutoTokenizer
    """
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        name = spec[3:].split("/")[-1]
        return name, lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return spec, enc.encode


def read_lines(path: str) -> list[str]:
    """Read non-empty, NFC-normalized lines from a text file."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def analyze(lines: list[str], encode) -> dict:
    """Compute multiple fertility metrics across all lines.
    
    Returns a dict with:
      - tok_per_word:      tokens / whitespace-split words  (v0 metric, fixed)
      - tok_per_grapheme:  tokens / grapheme clusters
      - tok_per_byte:      tokens / UTF-8 bytes
      - tok_per_char:      tokens / characters
      - tok_per_sentence:  average tokens per sentence (for parallel corpora)
      - total_tokens:      total token count
      - total_sentences:   number of sentences
    """
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0
    total_chars = 0
    n_sentences = len(lines)

    for line in lines:
        # Note: we do NOT lowercase here. Lowercasing changes token count
        # and is asymmetric: it's a no-op for Hindi but changes English.
        # For a fair cross-language comparison, we keep original casing.
        tokens = encode(line)
        words = line.split()  # FIX: use split() not split(" ")
        
        total_tokens += len(tokens)
        total_words += len(words)
        total_graphemes += count_grapheme_clusters(line)
        total_bytes += len(line.encode("utf-8"))
        total_chars += len(line)

    return {
        "tok_per_word":     total_tokens / total_words if total_words else 0,
        "tok_per_grapheme": total_tokens / total_graphemes if total_graphemes else 0,
        "tok_per_byte":     total_tokens / total_bytes if total_bytes else 0,
        "tok_per_char":     total_tokens / total_chars if total_chars else 0,
        "tok_per_sentence": total_tokens / n_sentences if n_sentences else 0,
        "total_tokens":     total_tokens,
        "total_sentences":  n_sentences,
        "total_words":      total_words,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Corrected tokenizer fertility benchmark (v1)"
    )
    ap.add_argument(
        "--corpus", action="append", required=True,
        metavar="LANG=PATH",
        help="Language code and path, e.g. eng=corpus/eng.txt (repeatable)",
    )
    ap.add_argument(
        "--tokenizer", action="append", default=None,
        help="Tokenizer spec (repeatable). Default: gpt2",
    )
    ap.add_argument(
        "--output", default=None,
        help="Path to write CSV results (optional)",
    )
    args = ap.parse_args()

    tokenizer_specs = args.tokenizer or ["gpt2"]

    # Parse corpus specs
    corpora = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        corpora[lang] = read_lines(path)
        print(f"Loaded {lang}: {len(corpora[lang])} sentences from {path}")

    print()

    all_results = []

    for tok_spec in tokenizer_specs:
        tok_name, encode = load_tokenizer(tok_spec)
        print(f"═══ Tokenizer: {tok_name} ═══")
        print(f"{'lang':<8} {'tok/word':>10} {'tok/grapheme':>13} "
              f"{'tok/byte':>10} {'tok/sent':>10} {'total_tok':>10}")
        print("─" * 70)

        tok_results = {}
        for lang, lines in corpora.items():
            metrics = analyze(lines, encode)
            tok_results[lang] = metrics
            all_results.append({
                "tokenizer": tok_name,
                "lang": lang,
                **metrics,
            })
            print(f"{lang:<8} {metrics['tok_per_word']:>10.3f} "
                  f"{metrics['tok_per_grapheme']:>13.3f} "
                  f"{metrics['tok_per_byte']:>10.4f} "
                  f"{metrics['tok_per_sentence']:>10.1f} "
                  f"{metrics['total_tokens']:>10d}")

        # Cross-language ratios (relative to first language)
        if len(tok_results) >= 2:
            langs = list(tok_results)
            base = langs[0]
            print()
            for lang in langs[1:]:
                r_word = tok_results[lang]["tok_per_word"] / tok_results[base]["tok_per_word"]
                r_sent = tok_results[lang]["tok_per_sentence"] / tok_results[base]["tok_per_sentence"]
                r_byte = tok_results[lang]["tok_per_byte"] / tok_results[base]["tok_per_byte"]
                print(f"  {lang} vs {base}:")
                print(f"    tok/word ratio:     {r_word:.2f}× "
                      f"({'worse' if r_word > 1 else 'better'})")
                print(f"    tok/sentence ratio: {r_sent:.2f}× "
                      f"({'worse' if r_sent > 1 else 'better'})"
                      f"  ← THIS is the cost multiplier for parallel content")
                print(f"    tok/byte ratio:     {r_byte:.2f}× "
                      f"({'worse' if r_byte > 1 else 'better'})")
        print()

    # Write CSV if requested
    if args.output and all_results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
