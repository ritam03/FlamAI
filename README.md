# FlamAI AI Team Intern Assignment — Submission

## Author
Ritam Pal

## Desmos Visualization
**[Tokenizer Fertility Comparison](https://www.desmos.com/calculator/chdjtmw3n2)**  
Black dots: GPT-2 tok/sentence ratio vs English | Red dots: XLM-RoBERTa tok/sentence ratio | Red line: y=1 (English baseline)  
Languages: 1=Eng, 2=Hin, 3=Kan, 4=Tam, 5=Tel, 6=Ben

## Repository Structure

```
submission/
├── NOTEBOOK.md         # Chronological lab notebook (hypothesis → experiment → result)
├── AI_USAGE.md         # Honest AI usage log
├── README.md           # This file
├── partA/              # Tokenizer audit (50 pts)
│   ├── download_corpus.py      # A1: OPUS-100 corpus download script
│   ├── corpus/                 # A1: Downloaded eval corpus (6 languages)
│   ├── audit_bugs.py           # A2: Bug demonstration with measured evidence
│   ├── fertility_v1.py         # A3: Corrected fertility script (multi-tokenizer, multi-denominator)
│   ├── parallel_analysis.py    # A3: Parallel sentence-level cost analysis
│   ├── results/                # A3: Output CSV with all fertility metrics
│   └── A4_memo.md              # A4: Recommendation memo
├── partB/
│   └── calculations.md         # B1-B4: Full KV-cache arithmetic + throughput analysis
└── partC/
    └── memo.md                 # Decision memo: SFT recommendation
```

## Quick Start

```bash
# Install dependencies
pip install tiktoken datasets transformers regex

# A1: Download corpus (requires internet)
python partA/download_corpus.py

# A2: Run bug audit (uses starter_kit corpus)
python partA/audit_bugs.py

# A3: Run corrected fertility analysis
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

# A3: Parallel sentence analysis
python partA/parallel_analysis.py
```

## Key Findings Summary

### Part A: Tokenizer Audit

| Finding | Type | Impact |
|---|---|---|
| `split(" ")` creates empty words from double-spaces | Code bug | ~2% fertility deflation |
| `tok/word` is wrong metric for cost comparison | Conceptual | Overstates/understates cost ratio depending on corpus |
| `line.lower()` on Hindi | Red herring (fine) | None — Devanagari has no case |

**Corrected headline:** With GPT-2, Hindi costs 6.44× English (per sentence). With XLM-RoBERTa, only 1.28×. The tokenizer choice matters more than the language.

### Part B: Capacity Reconciliation

- KV cache: 112 KiB/token → max ~27 concurrent 4096-token sequences on L4
- Throughput anomaly: KV cache preemption at batch ≥ 32 drops throughput by 14-19%
- Report's error: `reported_tok_s` includes prefill tokens. Real goodput at batch 24: ~201 tok/s (not 1607)
- The "3200 tok/s at batch 48" extrapolation is doubly wrong

### Part C: Decision Memo

Recommends SFT on synthetic casualized pairs with a day-1 prompt-engineering baseline. Kill criterion: <50% preference rate after week-1 Hindi pilot.
