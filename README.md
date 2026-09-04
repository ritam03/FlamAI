# 🔍 The Audit — FlamAI AI Team Intern Assignment

**Author:** Ritam Pal  
**Desmos:** [Tokenizer Fertility Visualization](https://www.desmos.com/calculator/chdjtmw3n2)

---

## TL;DR — The Three Claims I'll Defend to the Death

| # | Claim | Evidence | Impact |
|---|---|---|---|
| 1 | `split(" ")` is a bug — double-spaces in both corpora create phantom empty words | `audit_bugs.py` → +1.4% (eng), +2.0% (hin) fertility deflation | Small but real; direction: deflates fertility |
| 2 | **tok/word is the wrong metric** — it doesn't hold *meaning* constant across languages | `parallel_analysis.py` on 1,930 parallel pairs → tok/word says 5.43×, tok/sentence says 6.44× (GPT-2) | The *entire* framing of the cost comparison is wrong |
| 3 | `reported_tok_s` in bench_log includes **prefill tokens** — the report's "1311 tok/s" and "~3200 tok/s" numbers are fiction | Two independent derivations → real goodput at batch-24: **~201 tok/s** | Capacity plan is off by **8×** |

**What I deliberately did NOT flag:** `line.lower()` — it's a no-op on Devanagari (Hindi has no case). Looks suspicious, is harmless. Flagging it without evidence would cost -5 pts.

---

## 📊 The Number That Changes Everything

```
                    tok/word (what v0 reports)     tok/sentence (what actually matters)
                    ─────────────────────────     ──────────────────────────────────────
  GPT-2:            Hindi is 5.43× English        Hindi is 6.44× English
  XLM-RoBERTa:      Hindi is 1.08× English        Hindi is 1.28× English
                                                                    ↑
                                              The tokenizer choice eliminates 80% of the gap.
                                              Budget 1.3×, not 6×.
```

> **The single number for routing decisions:** `tok/sentence` ratio on a parallel eval corpus with your production tokenizer. This holds meaning constant — the only fair denominator for a cost comparison.

---

## 🗂 Repository Structure

```
.
├── NOTEBOOK.md              # Chronological lab notebook — dead ends included
├── AI_USAGE.md              # Where AI helped, where it misled me
├── README.md                # You are here
│
├── partA/                   # Tokenizer Audit (50 pts)
│   ├── audit_bugs.py        # A2: Isolates each bug, measures distortion, identifies red herring
│   ├── fertility_v1.py      # A3: Fixed script — multi-tokenizer × multi-denominator
│   ├── parallel_analysis.py # A3: Per-sentence cost on 1,930 parallel eng↔hin pairs
│   ├── download_corpus.py   # A1: OPUS-100 downloader (6 languages, reproducible)
│   ├── A4_memo.md           # A4: ≤1 page recommendation
│   ├── corpus/              # A1: 6 language files (eng, hin, kan, tam, tel, ben)
│   └── results/             # fertility_results.csv (12 rows: 2 tokenizers × 6 langs)
│
├── partB/
│   └── calculations.md      # B1–B4: KV-cache arithmetic, anomaly, misreading, metric
│
└── partC/
    └── memo.md              # Decision memo: SFT + day-1 experiment design
```

---

## ⚡ Reproduce Everything in 60 Seconds

```bash
pip install tiktoken datasets transformers regex

# A1 — Build the eval corpus (downloads OPUS-100 test splits)
python partA/download_corpus.py

# A2 — Run the bug audit (evidence for all three findings)
python partA/audit_bugs.py

# A3 — Corrected multi-tokenizer, multi-denominator analysis
python partA/fertility_v1.py \
    --corpus eng=partA/corpus/eng.txt \
    --corpus hin=partA/corpus/hin.txt \
    --corpus kan=partA/corpus/kan.txt \
    --corpus tam=partA/corpus/tam.txt \
    --corpus tel=partA/corpus/tel.txt \
    --corpus ben=partA/corpus/ben.txt \
    --tokenizer gpt2 \
    --tokenizer hf:xlm-roberta-base \
    --output partA/results/fertility_results.csv

# A3 — Parallel sentence cost comparison (the key insight)
python partA/parallel_analysis.py
```

Every script is self-contained, prints its own evidence, and exits cleanly.

---

## Part A — Tokenizer Audit (50 pts)

### A1: Corpus (10 pts)

**Source:** [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) test splits  
**Languages:** English, Hindi, Kannada (Dravidian), Tamil (Dravidian), Telugu, Bengali  
**Size:** 918–2,000 sentences per language (2,000 for most; 918 for Kannada due to smaller OPUS subset)

**Caveats I documented:**
- Domain bias: OPUS is formal/written (news, subtitles, Wikipedia) — not conversational
- The English base sentences differ across language pairs (not a single parallel set for all 6)
- 2,000 sentences: stable for mean estimates (CLT), but misses distributional tails
- No code-mixed text (Hindi+English), which is common in real Indian user traffic

### A2: Script Audit (20 pts)

| Finding | Type | How I measured it | Magnitude |
|---|---|---|---|
| `line.split(" ")` creates empty strings from double-spaces in both corpus files | **Code bug** | Compared `split(" ")` vs `split()` word counts on affected lines; recomputed fertility both ways | eng: +1.41%, hin: +2.01% (fertility deflated) |
| `tok/word` is the wrong denominator for cross-language *cost* comparison | **Conceptual** | Computed tok/sentence on 1,930 parallel pairs; compared with tok/word ratio | tok/word: 5.43×, tok/sentence: 6.44× (direction depends on corpus word-count asymmetry) |
| `line.lower()` on Hindi text | **Red herring — FINE** | Verified `.lower()` is a no-op on all Hindi test sentences (Devanagari has no case); measured 0-token delta | Zero impact. Defensible normalization for English. |

**Also fine:** `random.seed(1337)` (unused, no effect), `unicodedata.normalize("NFC")` (correct for Indic scripts).

### A3: Corrected Analysis (12 pts)

Ran `fertility_v1.py` with **2 tokenizers** × **4 denominators** × **6 languages**:

| Tokenizer | Metric | eng | hin | kan | tam | tel | ben |
|---|---|---|---|---|---|---|---|
| GPT-2 | tok/word | 1.30 | 7.08 | 15.57 | 21.67 | 16.17 | 11.33 |
| GPT-2 | tok/sentence | 16.1 | 103.3 | 52.7 | 126.0 | 61.9 | 134.7 |
| GPT-2 | tok/byte | 0.241 | 0.591 | 0.904 | 0.978 | 0.962 | 0.745 |
| XLM-R | tok/word | 1.45 | 1.56 | 2.71 | 2.52 | 2.34 | 1.97 |
| XLM-R | tok/sentence | 17.9 | 22.8 | 9.2 | 14.7 | 9.0 | 23.4 |
| XLM-R | tok/byte | 0.267 | 0.130 | 0.157 | 0.114 | 0.139 | 0.129 |

**Which number drives routing?** → `tok/sentence` with the production tokenizer. It holds meaning constant across languages — the only fair basis for a cost comparison.

### A4: Recommendation (8 pts)

→ See [`partA/A4_memo.md`](partA/A4_memo.md)

Core recommendation: deploy a multilingual tokenizer (XLM-R family or Indic-trained). Hindi drops from 6.4× to 1.3× cost. Monitor `median_tok_per_request` ratio in production.

---

## Part B — Capacity Reconciliation (20 pts)

→ Full derivations in [`partB/calculations.md`](partB/calculations.md)

### B1: KV-Cache Arithmetic

```
KV bytes/token = 2 (K+V) × 8 (kv_heads) × 128 (head_dim) × 28 (layers) × 2 (fp16)
               = 114,688 bytes = 112 KiB/token

Available KV memory = 24GB × 0.92 − 8.4GB (weights) − 1.6GB (overhead) ≈ 12.08 GB
Max 4096-token seqs = 12.08 GB / (112 KiB × 4096) ≈ 27 sequences
```

**Validated:** batch-24 fits (kv_util=0.93, 0 preemptions), batch-32 overflows (7 preemptions) ✓

### B2: Throughput Anomaly

Throughput **drops** at batch ≥ 32 (1607 → 1384 → 1299 tok/s) due to **KV cache preemption**. Evicted sequences must re-prefill at ~500ms each → wasted compute. Fix: cap `max_num_seqs=24` for long-context workloads.

### B3: The Report's Fatal Misreading

`reported_tok_s` counts **prefill + decode tokens**. Real generation goodput:

```
Method 1:  24 × 512 gen_tokens / 61.16s  = 200.9 tok/s
Method 2:  (1607.4 × 61.16 − 24 × 3584) / 61.16 = 201.0 tok/s
                                                     ↑
                                            Both methods agree.
                                            The report says 1607. Reality: 201.
```

### B4: Confirming Metric

Pull `num_preempted_sequences` from vLLM `/metrics`. Expect 0 at batch ≤ 24, 7 at batch 32.

---

## Part C — Decision Memo (15 pts)

→ See [`partC/memo.md`](partC/memo.md)

**Recommendation:** SFT on synthetic casualized pairs, preceded by a day-1 prompt-engineering baseline.

| Element | Detail |
|---|---|
| Data | 30K pairs (5K × 6 langs), ~9h GPU time to generate |
| Training | LoRA fine-tune, ~17 min on A100-80GB |
| Reviewer | 900 pairs validated in 30h (2 min/pair) |
| Success metric | ≥70% "appropriately casual" on blind A/B |
| Kill criterion | <50% after week-1 Hindi pilot → pivot to rewriter model |
| Day-1 experiment | Prompt-engineering baseline on 50 Hindi examples + start 5K pair generation |

---

## 📓 Notebook & AI Usage

- **[NOTEBOOK.md](NOTEBOOK.md)** — Chronological log with 6 documented dead ends / surprises
- **[AI_USAGE.md](AI_USAGE.md)** — Claude wrote ~70% of code, ~50% of prose. Three documented instances where it was wrong (FLORES gated access, NFC hallucination, metric direction).
