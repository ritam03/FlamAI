#!/usr/bin/env python3
"""
download_corpus.py — Download parallel multilingual eval corpus from OPUS-100.

Downloads English-aligned parallel sentences for Hindi, Kannada, Tamil,
Telugu, Bengali, and Malayalam from Helsinki-NLP/opus-100.

We use the test split (2000 sentences per language pair), which provides
a proper-sized eval set far exceeding the ~10 sentence toy corpus in
starter_kit/corpus_sample/.

Choice rationale:
  - OPUS-100: ungated, parallel, covers all required languages
  - Test split: 2000 sentences = large enough for stable fertility
    estimates (Central Limit Theorem), small enough for fast experiments
  - English-centric pairing: all languages aligned to English, enabling
    per-sentence cost comparison on identical semantic content.

Caveats (documented per A1 requirements):
  - Domain: OPUS-100 is sourced from multiple OPUS sub-corpora (Europarl,
    OpenSubtitles, WikiMatrix, etc.) — biased toward formal/written text,
    underrepresents colloquial/conversational register.
  - Parallel ≠ identical: translations may differ in length/style even
    when expressing the "same" meaning.
  - Sample size: 2000 sentences is adequate for mean estimation but may
    miss tail behavior (very long/short sentences, rare scripts).
  - Missing: no code-mixed text, no domain-specific jargon (legal, medical).

Usage:
    python download_corpus.py
"""

import os
import sys
import unicodedata


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install: pip install datasets")
        sys.exit(1)

    # OPUS-100 language pair configs (all English-centric)
    lang_pairs = {
        "eng": ("en-hi", "en"),     # English (from en-hi pair)
        "hin": ("en-hi", "hi"),     # Hindi
        "kan": ("en-kn", "kn"),     # Kannada (Dravidian)
        "tam": ("en-ta", "ta"),     # Tamil (Dravidian)
        "tel": ("en-te", "te"),     # Telugu (Dravidian — bonus)
        "ben": ("bn-en", "bn"),     # Bengali (bonus)
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
    os.makedirs(out_dir, exist_ok=True)

    # We need the parallel alignment — all sentences should correspond
    # to the same English sentences. Since OPUS-100 pairs are different
    # per language pair, we extract each pair's test split separately.
    
    # For truly parallel comparison, we use the en-hi pair as the baseline
    # (English + Hindi from the same 2000-sentence set) and then download
    # the other languages from their respective pairs.
    
    print("=" * 60)
    print("Downloading OPUS-100 test splits for multilingual eval")
    print("=" * 60)

    stats = {}

    for short_code, (config, lang_key) in lang_pairs.items():
        print(f"\n  Loading {short_code} from opus-100/{config}...")
        ds = load_dataset("Helsinki-NLP/opus-100", config, split="test")
        
        out_path = os.path.join(out_dir, f"{short_code}.txt")
        count = 0
        total_chars = 0
        total_words = 0
        
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                sent = row["translation"][lang_key].strip()
                if not sent:
                    continue
                sent = unicodedata.normalize("NFC", sent)
                f.write(sent + "\n")
                count += 1
                total_chars += len(sent)
                total_words += len(sent.split())

        stats[short_code] = {
            "sentences": count,
            "avg_words": total_words / count if count else 0,
            "avg_chars": total_chars / count if count else 0,
        }
        print(f"    → {count} sentences, avg {stats[short_code]['avg_words']:.1f} words/sent, "
              f"avg {stats[short_code]['avg_chars']:.1f} chars/sent")
        print(f"    → saved to {out_path}")

    # Also save the parallel English-Hindi file for sentence-level comparison
    print(f"\n  Creating parallel eng-hin pairs file...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="test")
    parallel_path = os.path.join(out_dir, "parallel_eng_hin.tsv")
    with open(parallel_path, "w", encoding="utf-8") as f:
        f.write("eng\thin\n")
        for row in ds:
            eng = unicodedata.normalize("NFC", row["translation"]["en"].strip())
            hin = unicodedata.normalize("NFC", row["translation"]["hi"].strip())
            if eng and hin:
                f.write(f"{eng}\t{hin}\n")
    print(f"    → saved to {parallel_path}")

    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(f"\n{'lang':<6} {'sentences':>10} {'avg_words':>10} {'avg_chars':>10}")
    print("─" * 40)
    for lang, s in stats.items():
        print(f"{lang:<6} {s['sentences']:>10} {s['avg_words']:>10.1f} {s['avg_chars']:>10.1f}")
    
    print(f"\nSource: Helsinki-NLP/opus-100 (test split)")
    print(f"Total languages: {len(stats)}")
    print(f"Output directory: {out_dir}")
    print("\nCAVEATS:")
    print("  - Domain bias: OPUS sourced from formal/written text (news, subtitles, wiki)")
    print("  - Not all languages share the same parallel English set")
    print("  - 2000 sentences: adequate for mean fertility, may miss distributional tails")
    print("  - No code-mixed or conversational text represented")


if __name__ == "__main__":
    main()
