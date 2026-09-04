# A4 — Recommendation Memo

## Corrected Headline Numbers

| Tokenizer | Hindi/English tok/word | Hindi/English tok/sentence | 
|---|---|---|
| GPT-2 (English-centric) | 5.43× | **6.44×** |
| XLM-RoBERTa (multilingual) | 1.08× | **1.28×** |

The v0 report's claim of "5.89× worse" was computed with tok/word on a 10-sentence toy corpus using the GPT-2 tokenizer. On our 2000-sentence OPUS-100 eval corpus:

- **With GPT-2:** The true cost ratio (tok/sentence) is **6.44×** — actually *higher* than reported, because tok/word understates the gap on this larger corpus (Hindi sentences in OPUS have proportionally more words than the toy corpus).
- **With XLM-RoBERTa:** The cost ratio collapses to **1.28×** — Hindi costs only 28% more than English.

**The critical insight:** The tokenizer choice dominates the language factor. Switching to a multilingual tokenizer reduces the Hindi cost gap from 6× to 1.3×.

## Routing Recommendation

1. **Deploy a multilingual-aware tokenizer** (XLM-RoBERTa, Gemma, or similar Indic-trained model) for all Indic-language serving. This alone eliminates most of the cost differential.
2. **Do not** route Indic traffic to a separate model solely based on tokenizer fertility — the 6× cost claim from the v0 report is an artifact of using GPT-2, not a property of Hindi.
3. **If the production model must use GPT-2 tokenization** (e.g., the model was trained with it), then budget ~6.4× token cost for Hindi relative to English, but actively evaluate model migration to a multilingual-tokenizer-based model.

## Biggest Caveat

Our eval corpus (OPUS-100) is biased toward **formal/written text** (news, Wikipedia, subtitles). Conversational Hindi — which is what the product team cares about (see Part C) — may exhibit different fertility characteristics:
- Code-mixing (Hindi+English) is common in conversational Indic text and would shift the cost ratio
- Colloquial Hindi uses shorter sentences, which could change the tok/sentence ratio
- The parallel alignment quality in OPUS varies; imperfect translations inflate variance

## Production Monitoring Metric

**Monitor: `median_tok_per_request` ratio between Hindi and English requests**, measured weekly on production traffic.

Why this metric:
- Captures the *actual* cost ratio on real user queries (not synthetic benchmarks)
- Uses tok/request as the denominator (equivalent to tok/sentence: one meaning unit)
- Median is robust to outliers (very long/short queries)
- If this ratio drifts significantly from our benchmark (1.28× for multilingual tokenizer), it signals either a change in user behavior or a data distribution shift that invalidates our analysis

**Alert threshold:** if the ratio exceeds 2.0× for any Indic language for >7 consecutive days, re-run the fertility analysis on a sample of production queries.
