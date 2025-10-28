# Evaluation System Notes

## Reconstruction Quality Metric Issue

### Current Implementation
The "reconstruction_quality" metric in baseline evaluation computes BLEU score between:
- **Reference**: Original full token sequence
- **Candidate**: Compressed token sequence (subset of original)

**Code location**: `evaluation/baselines.py:449`
```python
bleu_score = quality_metrics.compute_bleu(tokens, compressed_tokens)
```

### Why This is Wrong

This metric doesn't measure reconstruction quality at all. It measures how similar the compressed sequence is to the original, which will ALWAYS be low because:
1. The compressed sequence is a subset (e.g., 30% of tokens)
2. BLEU is designed for comparing sequences of similar length
3. No actual reconstruction is happening

### Expected Results
With 30% token retention:
- Maximum theoretical BLEU: ~0.3 (if tokens perfectly overlap)
- Typical BLEU: 0.01-0.10 (what we're seeing)

This explains why all baselines show 1-9% "reconstruction quality" - it's not measuring what we think it's measuring.

### What Should Happen

**True reconstruction quality evaluation:**
1. Compress sequence using policy (keep X% of tokens)
2. Pass compressed sequence through GPT-2 reconstructor
3. Compare reconstructed sequence to original using:
   - BLEU score
   - ROUGE score
   - Perplexity
   - Token-level accuracy

**Example flow:**
```
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Compressed: [1, 3, 5, 7, 9]  (50% retention)
Reconstructed: [1, 2, 3, 4, 5, 7, 8, 9, 10]  (GPT-2 fills in gaps)
Quality: BLEU(original, reconstructed) = 0.85
```

### Impact on Current Results

The baseline evaluation results are valid for what they measure (compression ratio), but the "reconstruction_quality" numbers are meaningless without the actual reconstructor.

This means:
1. We can't claim anything about reconstruction quality yet
2. The plots showing "quality" are misleading
3. Need to run full evaluation with reconstructor to get real quality metrics

### Fix Required

Update baseline evaluation to:
1. Load GPT-2 reconstructor model
2. Actually run reconstruction step
3. Compute quality metrics on reconstructed sequences

**OR** (simpler):

Just report compression ratios for baselines and clearly state that reconstruction quality requires the full pipeline with trained models.

### Updated Evaluation Plan

**Phase 1 (Current):** ✅
- Baseline compression ratios only
- No reconstruction claims

**Phase 2 (Needs trained RL agent):**
- Train RL agent policy
- Evaluate RL agent compression ratios vs baselines
- Show RL agent learns better compression

**Phase 3 (Full pipeline):**
- Run full pipeline: compress → reconstruct
- Measure true reconstruction quality
- Compare quality across all methods
- Measure inference speed

We're at Phase 1. The README has been updated to reflect this reality.
