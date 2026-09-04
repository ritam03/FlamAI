# NOTEBOOK.md — Chronological Lab Notebook

## Hour 0: Initial read-through (30 min)

Read the assignment twice. Key observations:
- This is an audit, not a build. They want me to find problems, not create solutions from scratch.
- The evidence rule is the hard constraint: every claim needs measured evidence.
- The "looks suspicious but is actually fine" trap in A2 is critical — I need to NOT flag something.
- Part B is arithmetic-heavy. Need to be precise.
- Part C is about reasoning under constraints, not about technical depth.

**Hypothesis going in:** The intern's report probably has both obvious code bugs and subtle conceptual errors. The fertility metric (tok/word) might not be the right thing to optimize for cost.

## Hour 1: Starter kit exploration (30 min)

### Examined fertility.py

Line-by-line read. Noted:
1. `split(" ")` on line 62 — immediately suspicious. Does the corpus have double spaces?
2. Checked corpus files: YES — `eng_sample.txt` line 7 has "books  in" and `hin_sample.txt` line 10 has "किताबें  अलमारी". Bug confirmed.
3. `line.lower()` on line 60 — my first instinct was "this is wrong for Hindi!" But then I realized: Hindi has no case distinction. `.lower()` is a no-op on Devanagari. This is the red herring.
4. `random.seed(1337)` — set but never used anywhere. Leftover from an earlier version. Harmless.
5. The bigger issue: the `analyze()` function computes tok/word, but *what should we actually compute?*

### Read REPORT_v0.md

The report claims "Hindi fertility is 5.89× worse than English" and recommends budgeting 6× serving cost. But:
- The denominator (words) isn't held constant across languages
- Hindi uses fewer words to express the same meaning
- The right comparison is tokens per *unit of meaning* (i.e., per sentence in a parallel corpus)

**Revised hypothesis:** Bug 1 (split) is small but real. Bug 2 (wrong metric) is the big conceptual error. The lower() is the red herring.

### Read bench/ files

- FLM-4B-Instruct: 4.2B params, GQA with 8 KV heads, fp16, on L4
- The bench_log.csv has an obvious inflection point: throughput drops at batch 32 while preempted_seqs jumps to 7
- The report's Section 2 conclusion about "longer prompts = better throughput" smells wrong — need to check if `reported_tok_s` includes prefill tokens

## Hour 2: Bug demonstration and evidence (45 min)

### Bug 1 — split(" ") vs split()

Wrote `audit_bugs.py` to isolate and measure:
- `split(" ")` on "Please keep the books  in the cupboard." produces 8 "words" (including one empty string)
- `split()` produces 7 words (correct)
- Effect on fertility: deflates by ~1.4% (English) and ~2.0% (Hindi)
- Small but measurable. Direction: inflates word count → deflates fertility

### Bug 2 — Conceptual: tok/word is wrong

This is the big one. Ran parallel analysis on the 10-sentence starter_kit corpus:
- v0 reports: 6.11× (tok/word ratio)
- True cost: 4.78× (tok/sentence ratio on same 10 parallel pairs)
- The difference comes from Hindi using 0.78× as many words per sentence for the same meaning

**Dead end #1:** Initially I thought the lower() issue was Bug 2. Spent 15 minutes verifying before realizing it's a no-op on Hindi. This was a productive dead end — it confirmed the red herring.

### Red herring — lower()

Measured token count with/without lowercasing:
- Hindi: 0 delta on all test sentences (confirmed no-op)
- English: varies by 0-1 tokens per sentence (capitalized words like "NASA" tokenize differently)
- Verdict: asymmetric but *defensible* — removing casing noise from English is reasonable, and the no-op on Hindi means the comparison isn't biased

## Hour 3: Corpus construction (40 min)

### Attempt 1: FLORES-200 — FAILED

Tried `facebook/flores` and `openlanguagedata/flores_plus` — both gated on HuggingFace, require authentication. Documented as a dead end.

### Attempt 2: OPUS-100 — SUCCESS

Used `Helsinki-NLP/opus-100` test splits:
- Available for all required languages: en-hi, en-kn, en-ta, en-te, bn-en
- Test split: ~2000 sentences per pair (918 for Kannada — smaller dataset)
- Parallel English-Hindi pairs enable sentence-level cost comparison

**Surprise:** Kannada only has 918 test sentences (vs 2000 for others). The OPUS-100 Kannada subset is much smaller. Need to note this as a caveat.

### Corpus caveats documented:
1. Domain bias: formal/written text (news, subtitles, Wikipedia)
2. Different English sets per language pair (not all 6 languages aligned to the same English sentences)
3. 2000 sentences: adequate for mean estimation, may miss tails
4. No code-mixed or conversational text

## Hour 4: Corrected analysis (1 hour)

### Multi-tokenizer fertility analysis

Ran `fertility_v1.py` with GPT-2 and XLM-RoBERTa on all 6 languages.

**Key surprise:** The XLM-RoBERTa results are dramatic:
- Hindi tok/word: 1.08× (vs GPT-2's 5.43×)
- Hindi tok/sentence: 1.28× (vs GPT-2's 6.44×)
- Tamil tok/sentence: 0.82× — actually *better* than English!

This means the v0 report's "6× cost" claim is entirely an artifact of using GPT-2 tokenizer. A multilingual tokenizer nearly eliminates the gap.

### Parallel sentence analysis

Ran `parallel_analysis.py` on 1930 eng-hin parallel pairs:
- GPT-2: true cost ratio = 6.44× (tok/sentence)
- XLM-R: true cost ratio = 1.28× (tok/sentence)
- Distribution analysis: GPT-2 has high variance (stdev 3.51), XLM-R has lower (1.79)

**Revised conclusion:** The v0 report's tok/word metric happens to understate the GPT-2 cost gap on this corpus (5.43× vs 6.44× true). But the REAL fix is switching tokenizers, not routing traffic.

## Hour 5: Part B — KV cache arithmetic (45 min)

### B1: KV-cache bytes per token

Straightforward calculation: 2 × 8 × 128 × 28 × 2 = 114,688 bytes = 112 KiB per token.

Max concurrent 4096-token sequences: ~27 (calculated from available GPU memory after subtracting model weights and runtime overhead).

**Validated against log:** batch 24 shows kv_cache_util=0.93 (24/27=0.89, close) and 0 preemptions. Batch 32 shows preemptions, confirming capacity ≈ 27.

### B2: Throughput anomaly

The long-context sweep shows throughput peaking at batch 24 (1607 tok/s) and DROPPING at batch 32 (1384) and 48 (1299). Mechanism: KV cache preemption — at batch 32, 7 sequences are evicted and must be re-prefilled.

### B3: Report's misreading

**This was the most interesting find.** The report compares `reported_tok_s` across prompt lengths and concludes "longer prompts give better throughput." But `reported_tok_s` includes prefill tokens!

Computed honest goodput for batch-24 long: (24 × 512) / 61.16 = 200.9 tok/s. Two independent methods agree.

The batch-48 extrapolation to "~3200 tok/s" is doubly wrong: wrong metric AND ignores preemption.

### B4: Confirming metric

`num_preempted_sequences` from vLLM's metrics endpoint. Expected: 0 for batch ≤ 24, 7 for batch 32, 23 for batch 48.

## Hour 6: Part C and writeup (1 hour)

### Decision memo

Chose SFT approach. Key arithmetic:
- Data gen: ~9 hours GPU time for 43K raw pairs
- Training: ~17 minutes for LoRA on 30K examples  
- Reviewer: 900 pairs validated in 30 hours
- Kill criterion: < 50% preference rate after week-1 Hindi pilot

### NOTEBOOK and AI_USAGE

Writing these now. The notebook is intentionally chronological with dead ends preserved.

## Summary of dead ends and surprises:

1. **Dead end:** Initially flagged `lower()` as a bug before verifying it's a no-op on Hindi → red herring
2. **Dead end:** Tried FLORES-200 — gated on HuggingFace → pivoted to OPUS-100
3. **Surprise:** Kannada OPUS-100 only has 918 sentences (vs 2000 for others)
4. **Surprise:** XLM-RoBERTa makes Hindi 1.28× English cost (not 6×!)
5. **Surprise:** The tok/word metric actually *understates* GPT-2's Hindi cost on OPUS (5.43× vs 6.44× tok/sentence) — opposite of what I expected from the toy corpus analysis
6. **Surprise:** The v0 report's Section 2 error is not just about long-vs-short comparison — `reported_tok_s` fundamentally counts the wrong thing for capacity planning
