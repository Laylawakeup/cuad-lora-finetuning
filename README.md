# CUAD Clause Classification — LoRA Fine-tuning Experiment

## Overview
Fine-tuned Qwen2.5-1.5B-Instruct on the CUAD legal contract dataset using LoRA,
targeting binary clause classification (Yes/No) across 6 contract clause categories.

---

## Task Definition
Given a contract clause, determine whether it contains a specific legal provision.

**Target categories:**
- Governing Law
- Termination for Convenience
- Liquidated Damages
- Non-Compete
- Limitation of Liability
- Arbitration

**Input format:**
Instruction:
Review the following contract clause and identify whether it contains
a 'Termination for Convenience' provision.

Contract Clause:
Either party may terminate this Agreement at any time upon 30 days
written notice to the other party...

Response:
Yes. The clause contains a 'Termination for Convenience' provision.
Relevant excerpt: "..."

---
## Dataset
- **Source:** [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad) — 510 commercial contracts, SQuAD-style annotations
- **Raw samples:** 2,550 (contract × category pairs)
- **Class distribution:** Yes 34.4% / No 65.6%
- **Split:** 80% train / 10% val / 10% test
### Data Processing Decisions
1. **Format conversion:** SQuAD-style QA → instruction-tuning format.
   Answers present = Yes + excerpt; no answers = No.
2. **Truncation:** Clause text truncated to 1,200 characters to fit within
   token budget. Known limitation — see Error Analysis.
3. **Class balancing:** Oversampled Yes class to 1:1 ratio for Experiment 2
   after Experiment 1 showed majority-class collapse.
---
## Training Configuration
| Config | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| Model | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Method | LoRA (Unsloth) | LoRA (Unsloth) |
| Rank | 8 | 16 |
| LoRA Alpha | 16 | 32 |
| Learning Rate | 2e-4 | 5e-5 |
| Epochs | 3 | 3 |
| Batch Size | 16 (4 × 4 accum) | 16 (4 × 4 accum) |
| Train Data | Unbalanced (65% No) | Oversampled 1:1 |
---
## Results
| Metric | Base Model (zero-shot) | Exp 1 (r=8, unbalanced) | Exp 2 (r=16, balanced) |
|--------|----------------------|------------------------|------------------------|
| Accuracy | 0.60 | 0.64 | **0.77** |
| F1 Yes | 0.39 | 0.12 | **0.69** |
| F1 No | 0.71 | 0.78 | **0.82** |
| Macro F1 | 0.55 | 0.45 | **0.75** |


**Key finding:**
Class imbalance was the primary failure mode.
Experiment 1 achieved higher accuracy than baseline by collapsing to
majority-class prediction (Yes recall = 0.07), making Macro F1 worse
than zero-shot. 
Oversampling in Experiment 2 recovered Yes F1 from
0.12 to 0.69, with Macro F1 improving from 0.55 (baseline) to 0.75.
All runs tracked in W&B:
`https://wandb.ai/laylawakeup-testing/cuad-lora-finetuning`
---
## Error Analysis
Three distinct failure modes identified on the test set:
**1. Reasoning Contradiction**
The model occasionally extracts correct evidence but outputs the wrong
label. In one Termination for Convenience case, the model quoted
*"may be terminated at any time without penalty"* in its response yet
predicted No — indicating a gap between evidence retrieval and final
classification.
**2. Context Truncation**
Clauses are truncated to 1,200 characters. For long contracts, the
relevant clause may fall outside this window, causing systematic false
negatives. A sliding-window or retrieval-augmented approach would
mitigate this.
**3. Superficial Pattern Matching**
The model over-triggers on surface-level keywords. In one false
positive, it classified a general termination clause as Termination for
Convenience upon seeing "terminated at any time upon 30 days' notice,"
without recognising that the clause contained conditions disqualifying
it as an unconditional convenience termination.

| Error Type | Root Cause | Potential Fix |
|------------|-----------|---------------|
| Reasoning Contradiction | Classification head disconnected from reasoning | Chain-of-thought training |
| Context Truncation | 1,200-char hard cutoff | Sliding window / RAG retrieval |
| Superficial Pattern Matching | Lack of deep legal semantics | Larger model / legal domain pre-training |
---
## Reproduction
```bash
# Install dependencies
pip install unsloth trl transformers datasets wandb huggingface_hub scikit-learn
# Run notebook
# 1. Data processing   
# 2. Baseline eval     
# 3. LoRA fine-tuning  
# 4. Eval
