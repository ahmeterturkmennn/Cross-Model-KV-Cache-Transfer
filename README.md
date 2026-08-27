# Cross-Model KV Cache Transfer — Implementation Notes

This repository contains a partial, exploratory implementation of the ideas from
**"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for
Prefill Reuse."** The goal of the paper is to check whether the KV cache produced by a
small model in a model family can be mapped, with a simple linear transform, onto the
KV cache of a larger model in the same family — which would allow reusing prefill
computation across model sizes.

This notebook is a first pass at reproducing that idea, **not** a faithful,
paper-parameter reproduction. See "Deviations from the paper" below before drawing any
conclusions from the numbers.

## Models

- Source (small) model: `Qwen/Qwen3-1.7B` — 28 layers, 8 KV heads, head_dim 128
- Target (large) model: `Qwen/Qwen3-4B` — 36 layers, 8 KV heads, head_dim 128

Both models are loaded via `transformers.AutoModelForCausalLM` with `torch_dtype="auto"`
and `device_map="auto"`.

## Data

Sequences are streamed from `HuggingFaceFW/fineweb-edu` (`default` split, `train`,
streaming mode), tokenized with the small model's tokenizer, filtered to sequences of at
least 1024 tokens, and truncated to exactly 1024 tokens. Sequences are collected until
the target sample count is reached (see below).

## Method

1. Run both models on the same input sequence with `use_cache=True` to obtain the full
   KV cache (`past_key_values`) for the source and target model.
2. Subsample the sequence dimension with a stride of 4 (`::4`) before fitting, to keep
   the regression problem tractable.
3. For a chosen `(source_layer, target_layer, source_head, target_head)` combination,
   flatten the sampled keys (or values) across all sequences into `X` (source) and `Y`
   (target).
4. Fit a linear map `Y ≈ X W + b` via least squares (`torch.linalg.lstsq` on the
   bias-augmented `X`), and, in a later cell, via ridge regression
   (`torch.linalg.solve` with an L2 penalty `alpha = 0.1` on the weight block only, bias
   left unregularized) as a more numerically stable alternative.
5. Evaluate the fit quality with the **R²** coefficient of determination between the
   predicted and actual target keys/values.
6. Repeat this for:
   - A single (layer, head) pair, as a sanity check.
   - Every (source_layer, target_layer) pair (mean R² across all 8 heads), producing a
     28×36 R² heatmap.
   - For a fixed (source_layer, target_layer) pair, every (source_head, target_head)
     combination, producing an 8×8 head-alignment R² matrix, from which the best
     source head is picked for each target head.

## Deviations from the paper

This implementation is **not** a strict reproduction. Known differences:

- **Sample size:** the paper's experiments use 500 sequences; this notebook uses **50**
  sequences (with one exploratory rerun at 100) due to local compute/time constraints.
  R² values here should be treated as noisier, lower-confidence estimates than the
  paper's reported numbers.
- **RoPE stripping:** the paper strips the rotary positional embedding (RoPE) from the
  keys before fitting the linear map, so that the regression targets purely
  content-dependent (position-independent) representations. **This notebook does not do
  that yet** — the keys used here still carry the RoPE rotation baked in. This is the
  main reason the filename/notes refer to this version as "unstripped." A RoPE-stripped
  version is planned as a follow-up (see below).
- Regularization (ridge) is only demonstrated on a single (layer, head) pair (cell with
  `alpha = 0.1`), not applied across the full layer/head sweep yet.

Because RoPE is not stripped, the fitted linear maps are entangling positional and
content information together, which likely understates how well a purely
content-based linear map could work, and may make the best-matching layer/head pairs
found here differ from what a RoPE-stripped analysis would find.

## Planned next steps

- Implement RoPE stripping (un-rotating the keys using each model's rotary frequencies
  before fitting) and rerun the layer-wise and head-wise R² sweeps for a fair
  comparison against the paper's setup.
- Scale the sample size back up toward the paper's 500 sequences once compute allows.
- Apply ridge regularization consistently across the full sweep, not just the single
  example pair.
- Validate the fitted maps by actually substituting the transformed cache into the
  target model's generation (not just measuring R² on the keys/values themselves).

## Notebook structure (cell-by-cell overview)

| Section | What it does |
|---|---|
| Model/tokenizer loading | Loads Qwen3-1.7B (source) and Qwen3-4B (target) |
| Load Data | Streams and filters 1024-token sequences from FineWeb-Edu |
| Examine Shapes | Inspects K/V cache shapes for a single layer/head |
| Example Linearity Check (single head) | Least-squares fit + R² for one (layer, head) pair, 20 sequences |
| Generating KV Caches (50 sequences) | Builds cached, strided K/V tensors for 50 sequences across all layers |
| Example Layer Comparison | Head-averaged R² for a fixed (source_layer=14, target_layer=14) |
| Layerwise Linearity Check | Full 28×36 mean-R² matrix across all source/target layer pairs, plotted as a heatmap |
| Best-match selection | Picks the best source layer for each target layer |
| Headwise Comparison | 8×8 R² matrix between source/target heads for one layer pair |
| Best head selection | Picks the best source head for each target head |
| Ridge regression example | Demonstrates a regularized fit for one (layer, head) pair |
| 100-sequence rerun | Repeats the cache generation and layer comparison with 100 sequences |

## Requirements

```
transformers
accelerate
datasets
torch
matplotlib
```

A GPU is strongly recommended — the notebook uses `device_map="auto"` and runs both a
1.7B and a 4B model.
