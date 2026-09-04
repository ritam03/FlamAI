# Part B — Capacity Reconciliation

## B1: KV-Cache Computation (7 pts)

### (a) KV-cache bytes per token

From `model_spec.md`:

| Parameter | Value |
|---|---|
| KV heads (GQA) | 8 |
| head_dim | 128 |
| layers | 28 |
| KV cache precision | fp16 (2 bytes) |

Each token requires storing both a **Key** and a **Value** vector for every layer:

```
bytes_per_token = 2 (K+V) × n_kv_heads × head_dim × n_layers × sizeof(fp16)
                = 2 × 8 × 128 × 28 × 2
                = 114,688 bytes
                = 112 KiB per token
```

### (b) Maximum concurrent 4096-token sequences

**GPU memory budget:**

```
Total GPU memory:          24 GB
Usable (gpu_mem_util=0.92): 24 × 0.92 = 22.08 GB

Model weights (fp16):      4.2B × 2 bytes = 8.4 GB
Runtime overhead:           ~1.6 GB (given)

Available for KV cache:    22.08 − 8.4 − 1.6 = 12.08 GB
                          = 12,969,164,800 bytes (using 12.08 × 1024³)
```

Wait, let me be precise:

```
22.08 GB = 22.08 × 1,073,741,824 = 23,709,016,064 bytes
 8.4  GB =  8.4  × 1,073,741,824 =  9,019,431,117 bytes
 1.6  GB =  1.6  × 1,073,741,824 =  1,717,986,918 bytes

Available = 23,709,016,064 − 9,019,431,117 − 1,717,986,918
          = 12,971,598,029 bytes
          ≈ 12.08 GiB
```

**Max concurrent sequences at 4096 tokens:**

```
KV per sequence = 114,688 bytes/token × 4,096 tokens = 469,762,048 bytes ≈ 448 MiB

max_sequences = 12,971,598,029 / 469,762,048 ≈ 27.6 → ~27 sequences
```

### Validation against bench_log.csv

| batch_size | prompt+gen | Total tokens | kv_cache_util | preempted |
|---|---|---|---|---|
| 24 | 3584+512=4096 | 4096 | 0.93 | 0 |
| 32 | 3584+512=4096 | 4096 | 0.97 | 7 |

- At batch 24: 24/27 = 0.89 predicted utilization — log shows 0.93 ✓ (close; the small discrepancy comes from block-level granularity in vLLM's paged attention)
- At batch 32: 32 > 27, so preemption is expected — log confirms 7 preempted sequences ✓
- This validates our calculation: the capacity ceiling is between 24 and 32, consistent with ~27.

---

## B2: Throughput Anomaly in Long-Context Sweep (6 pts)

### The data

| batch | prompt_len | gen_len | tok/s | ttft_ms | itl_ms | e2e_p95 | preempted | kv_util |
|---|---|---|---|---|---|---|---|---|
| 4 | 3584 | 512 | 565.4 | 483.2 | 51.33 | 32673 | 0 | 0.16 |
| 8 | 3584 | 512 | 902.6 | 519.0 | 62.26 | 39983 | 0 | 0.31 |
| 16 | 3584 | 512 | 1311.4 | 498.3 | 77.20 | 54602 | 0 | 0.62 |
| 24 | 3584 | 512 | 1607.4 | 500.5 | 96.07 | 69221 | 0 | 0.93 |
| **32** | 3584 | 512 | **1384.0** | **636.9** | 101.79 | 97466 | **7** | 0.97 |
| **48** | 3584 | 512 | **1298.5** | **955.4** | 100.00 | 105428 | **23** | 0.97 |

### The anomaly

Throughput should scale approximately linearly with batch size (as it does from batch 4→8→16→24). But at **batch 32, throughput drops from 1607 to 1384 tok/s** (−14%), and at **batch 48 it drops further to 1299 tok/s** (−19% vs batch 24).

### The mechanism: KV cache preemption

1. **KV cache saturation**: At batch 32, KV utilization hits 0.97 — the cache is nearly full.
2. **Preemption**: The scheduler cannot fit all 32 sequences simultaneously (capacity ≈ 27). It must **preempt** (evict) sequences: 7 at batch 32, 23 at batch 48.
3. **Wasted compute**: Preempted sequences must be **re-prefilled** when they're scheduled again. For a 3584-token prompt, re-prefill is expensive (~500ms per sequence based on TTFT). This wasted prefill compute directly reduces effective throughput.
4. **TTFT inflation**: TTFT jumps from ~500ms to 637ms (batch 32) and 955ms (batch 48), confirming re-prefill overhead.
5. **Wall-clock blowup**: batch 48 takes 151.4s vs 61.2s for batch 24, despite only 2× more requests — a clear sign of thrashing.

### Proposed config change

**Set `max_num_seqs=24`** (or more precisely, `max_num_seqs=27`) in the vLLM serving config for long-context workloads (prompt ≥ 2K tokens).

**Predicted quantitative effect**: 
- Prevents all preemption (preempted_seqs → 0)
- Maintains throughput at ~1607 tok/s instead of degrading  
- For 48 requests: they would be processed in 2 waves of 24, each taking ~61s → total ~122s vs 151s (19% faster)
- TTFT stays at ~500ms instead of inflating to 955ms

---

## B3: The Report's Misreading (4 pts)

### What the report says

> "at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization."
>
> "batch 48 should give us ~3200 tok/s"

### The misreading

The report treats `reported_tok_s` as **generation throughput**, but it is actually **total token throughput** — it counts both prefill (prompt) tokens and generated (decode) tokens.

- Short-prompt batch 16: each request processes 512 + 256 = 768 tokens total
- Long-prompt batch 16: each request processes 3584 + 512 = 4096 tokens total

Longer prompts have more prefill tokens counted in `reported_tok_s`, inflating the number. This does **not** mean the GPU is producing more useful output.

### Honest "goodput" of batch-24 long-prompt

**Method 1 — from wall clock:**

```
generated_tokens = num_requests × gen_len = 24 × 512 = 12,288
goodput = 12,288 / 61.16s = 200.9 tok/s
```

**Method 2 — from reported_tok_s:**

```
total_tokens_processed = reported_tok_s × wall_clock_s = 1607.4 × 61.16 = 98,276.6
prefill_tokens = num_requests × prompt_len = 24 × 3584 = 85,984 [* see note]
generated_tokens = 98,276.6 − 85,984 = 12,292.6
goodput = 12,292.6 / 61.16 = 201.0 tok/s
```

**Both methods agree: goodput ≈ 201 tok/s** (not 1607 tok/s).

*[Note: prefill tokens are processed in parallel and very fast, but they are still counted in reported_tok_s. The actual useful output is only the 512 generated tokens per request.]*

### What the report should have said

1. **Longer prompts do NOT give better GPU utilization for generation** — they just inflate the total-token counter with cheap prefill tokens. The actual decode throughput per sequence is similar.

2. **The batch-48 extrapolation to ~3200 tok/s is doubly wrong:**
   - It linearly extrapolates `reported_tok_s` (which includes prefill) — wrong metric
   - It ignores KV cache preemption: batch 48 actually *degrades* to 1299 tok/s (total), and the generation goodput is even lower: (48 × 512) / 151.41 = **162 tok/s** — *worse* than batch 24's 201 tok/s

3. **For capacity planning:** at batch 24 (max safe batch for long context), the system delivers ~201 tok/s of generated output. Scale hardware linearly from this number, not from 1607.

---

## B4: Confirming Metric (3 pts)

**Metric to pull:** `num_preempted_sequences` (or equivalently, `num_preemption` in vLLM's metrics endpoint at `/metrics`).

**Why:** This counter directly measures KV cache eviction events. It is the causal mechanism behind the throughput anomaly identified in B2.

**Expected values:**
- Batch ≤ 24: `num_preempted_sequences = 0` (all sequences fit in KV cache simultaneously)
- Batch 32: `num_preempted_sequences = 7` (matches log)
- Batch 48: `num_preempted_sequences = 23` (matches log)

If this counter is non-zero in production, it signals that the workload exceeds KV cache capacity and throughput is being wasted on re-prefilling evicted sequences. The operational response is to either reduce `max_num_seqs`, upgrade to more GPU memory, or enable prefix caching to reduce re-prefill cost.

**Secondary confirmation:** `gpu_kv_cache_usage_percent` — expect it to plateau at ~0.97 when preemption begins, confirming the cache is full rather than some other bottleneck (e.g., compute-bound).
