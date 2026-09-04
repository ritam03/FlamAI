# AI_USAGE.md — Honest AI Assistance Log

## Tools Used

- **Claude (Anthropic)** — primary assistant for code generation, analysis structuring, and writing
- **GitHub Copilot** — inline code completion (minor)

## Where AI Helped

### Code generation (~70% of code by volume)
- `download_corpus.py`: AI generated the initial OPUS-100 download script. I had to debug the FLORES-200 gated access issue and redirect to OPUS-100 myself.
- `fertility_v1.py`: AI scaffolded the multi-tokenizer, multi-denominator structure. I specified the metrics and the fix for `split()`.
- `audit_bugs.py`: AI helped structure the bug demonstration script. The identification of bugs was mine; the structured output formatting was AI-assisted.
- `parallel_analysis.py`: AI wrote most of this after I described what I needed (per-sentence token comparison on parallel corpus).

### Writing (~50% AI-assisted)
- Part B calculations: AI helped format the LaTeX-style arithmetic. I derived the numbers.
- Part C memo: AI helped structure the format (assumptions / arithmetic / success metric / kill criterion). The reasoning and specific numbers are mine.
- NOTEBOOK.md: AI helped with initial structure; the content reflects my actual workflow.

### Analysis (~30% AI-assisted)
- AI helped identify that `random.seed(1337)` is unused (I was looking for it as a potential bug).
- AI confirmed my intuition that `lower()` is a no-op on Devanagari.

## Where AI Misled Me

### False bug identification (caught and corrected)
- Claude initially suggested that `unicodedata.normalize("NFC")` might be problematic for some Indic scripts. After investigation, NFC is the correct normalization for Devanagari and other Indic scripts. This was a hallucination about Unicode edge cases.

### Overconfident metric claims (caught and corrected)  
- When computing the conceptual bug, AI initially framed tok/word as *always* overstating the cost gap. On the larger OPUS corpus, it turned out tok/word *understates* the gap (5.43× vs 6.44× tok/sentence). The direction depends on the word-count ratio of the corpus, which I had to verify empirically.

### FLORES-200 access
- AI confidently generated a download script for `facebook/flores` without noting it's a gated dataset. The script failed on first run. I had to research alternatives and switch to OPUS-100.

## What I Understand vs. What I Don't

### I understand deeply:
- Why tok/word is the wrong metric for cost comparison (denominator must hold meaning constant)
- The KV cache arithmetic in Part B (I've worked with transformer serving before)
- Why `reported_tok_s` includes prefill tokens and why that's misleading
- The preemption mechanism in vLLM and its effect on throughput

### I understand at a working level:
- Tokenizer internals (BPE merge rules, why GPT-2 is bad for Indic)
- GQA memory savings vs MHA
- The SFT recommendation in Part C (I understand the tradeoffs but haven't done large-scale SFT myself)

### I'm less confident about:
- Whether the OPUS-100 corpus is truly representative enough for the fertility analysis. The domain distribution matters and I haven't verified it thoroughly.
- The exact vLLM preemption behavior — I'm reasoning from the data patterns, but I haven't looked at vLLM source code to confirm the mechanism.
- Part C's reviewer throughput estimate (2 min/pair) — this could be optimistic for complex Hindi sentences.
