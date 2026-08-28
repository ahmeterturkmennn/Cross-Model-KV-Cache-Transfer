# Cross-Model KV Cache Transfer — Implementation Notes

This repository contains a partial, exploratory implementation of the ideas from
**"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for
Prefill Reuse."** The goal of the paper is to check whether the KV cache produced by a
small model in a model family can be mapped, with a simple linear transform, onto the
KV cache of a larger model in the same family — which would allow reusing prefill
computation across model sizes.

This repository currently contains **two versions of the implementation**:

1. An **unstripped RoPE version**, which is the more complete implementation and contains
   the main layer-wise and head-wise experiments.
2. A **RoPE-stripped version**, where the keys are un-rotated before regression. This
   version is currently **incomplete and should be considered experimental**; the full
   layer/head sweeps and downstream validation have not yet been reproduced with the
   stripped representations.

These notebooks are a first pass at reproducing the paper's idea, **not a faithful,
paper-parameter reproduction**. See "Deviations from the paper" below before drawing any
conclusions from the numbers.

## Models

* Source (small) model: `Qwen/Qwen3-1.7B` — 28 layers, 8 KV heads, head_dim 128
* Target (large) model: `Qwen/Qwen3-4B` — 36 layers, 8 KV heads, head_dim 128

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
   flatten the sampled keys (or values) across all sequences into `X` (source) and
   `Y` (target).
4. Fit a linear map `Y ≈ X W + b` via least squares (`torch.linalg.lstsq` on the
   bias-augmented `X`), and, in a later cell, via ridge regression
   (`torch.linalg.solve` with an L2 penalty `alpha = 0.1` on the weight block only, bias
   left unregularized) as a more numerically stable alternative.
5. Evaluate the fit quality with the **R²** coefficient of determination between the
   predicted and actual target keys/values.
6. Repeat this for:

   * A single (layer, head) pair, as a sanity check.
   * Every (source_layer, target_layer) pair (mean R² across all 8 heads), producing a
     28×36 R² heatmap.
   * For a fixed (source_layer, target_layer) pair, every (source_head, target_head)
     combination, producing an 8×8 head-alignment R² matrix, from which the best
     source head is picked for each target head.

### RoPE-stripped variant

A second notebook/version has also been uploaded that attempts to reproduce the paper's
**RoPE-stripping procedure**. In this version, the rotary positional component is removed
from the key representations before fitting the linear mappings.

However, this version is **not complete yet**. It should currently be treated as an
experimental work-in-progress rather than a complete reproduction. In particular, the
full layer-wise/head-wise analysis and the corresponding comparisons have not yet been
rerun using the stripped representations.

## Deviations from the paper

This implementation is **not a strict reproduction**. Known differences include:

* **Sample size:** the paper's experiments use 500 sequences; this notebook uses **50**
  sequences (with one exploratory rerun at 100) due to local compute/time constraints.
  R² values here should be treated as noisier, lower-confidence estimates than the
  paper's reported numbers.
* **RoPE handling:** the main/original notebook uses keys with RoPE still baked into
  them. A separate **RoPE-stripped notebook has now been uploaded**, but it is currently
  incomplete and does not yet reproduce the complete experimental pipeline.
* **Regularization:** ridge regression is demonstrated on a single (layer, head) pair
  (cell with `alpha = 0.1`), rather than being applied consistently across the full
  layer/head sweep.
* **Downstream validation:** the current experiments primarily evaluate linearity through
  R². The transformed KV cache has not yet been fully validated by substituting it into
  the target model during generation.

The distinction between the two versions is important: the unstripped notebook provides
the more complete exploratory results, while the RoPE-stripped notebook is intended to
move the implementation closer to the paper's methodology but is still unfinished.

## Planned next steps

* Complete the RoPE-stripped implementation and rerun the layer-wise and head-wise R²
  sweeps.
* Compare the unstripped and RoPE-stripped results directly.
* Scale the sample size back up toward the paper's 500 sequences once compute allows.
* Apply ridge regularization consistently across the full sweep, not just the single
  example pair.
* Validate the fitted maps by actually substituting the transformed cache into the
  target model's generation and measuring the effect on model outputs.
* Compare the resulting layer/head mappings against the mappings reported in the paper.

## Notebook structure (cell-by-cell overview)

### Main / Unstripped Notebook

| Section                               | What it does                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| Model/tokenizer loading               | Loads Qwen3-1.7B (source) and Qwen3-4B (target)                                      |
| Load Data                             | Streams and filters 1024-token sequences from FineWeb-Edu                            |
| Examine Shapes                        | Inspects K/V cache shapes for a single layer/head                                    |
| Example Linearity Check (single head) | Least-squares fit + R² for one (layer, head) pair, 20 sequences                      |
| Generating KV Caches (50 sequences)   | Builds cached, strided K/V tensors for 50 sequences across all layers                |
| Example Layer Comparison              | Head-averaged R² for a fixed (source_layer=14, target_layer=14)                      |
| Layerwise Linearity Check             | Full 28×36 mean-R² matrix across all source/target layer pairs, plotted as a heatmap |
| Best-match selection                  | Picks the best source layer for each target layer                                    |
| Headwise Comparison                   | 8×8 R² matrix between source/target heads for one layer pair                         |
| Best head selection                   | Picks the best source head for each target head                                      |
| Ridge regression example              | Demonstrates a regularized fit for one (layer, head) pair                            |
| 100-sequence rerun                    | Repeats the cache generation and layer comparison with 100 sequences                 |

### RoPE-Stripped Notebook

The second notebook contains the initial implementation for removing/un-rotating RoPE
from the key representations before regression. It is **incomplete** and is currently
intended as the next stage of the reproduction rather than as a finished experimental
result.

## Requirements

```text
transformers
accelerate
datasets
torch
matplotlib
```

A GPU is strongly recommended — the notebooks use `device_map="auto"` and run both a
1.7B and a 4B model.
