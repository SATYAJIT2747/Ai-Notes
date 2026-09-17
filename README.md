# LLM Efficiency & Scaling — Complete Study Notes

> A detailed study guide covering **Quantization**, **Attention Variants**, and **Scaling Laws**.
>
> Goal: understand not just *what* each technique does, but **why it exists, the mathematics behind it, the trade-offs, and how the three topics fit together**.

---

# Table of Contents

1. [Big Picture](#1-big-picture)
2. [Part I — Quantization](#2-part-i--quantization)
   - [Why Quantization?](#21-why-quantization)
   - [Bits and Number Formats](#22-bits-and-number-formats)
   - [Floating Point Numbers](#23-floating-point-numbers)
   - [FP32, FP16, BF16, FP8](#24-fp32-fp16-bf16-fp8)
   - [Integer Quantization](#25-integer-quantization)
   - [What Quantization Actually Does](#26-what-quantization-actually-does)
   - [Symmetric Quantization](#27-symmetric-quantization)
   - [Asymmetric Quantization](#28-asymmetric-quantization)
   - [Per-Tensor Quantization](#29-per-tensor-quantization)
   - [Per-Channel Quantization](#210-per-channel-quantization)
   - [Quantization Error](#211-quantization-error)
   - [Weights vs Activations vs KV Cache](#212-weights-vs-activations-vs-kv-cache)
   - [PTQ](#213-post-training-quantization-ptq)
   - [QAT](#214-quantization-aware-training-qat)
   - [Fake Quantization](#215-fake-quantization)
   - [Straight-Through Estimator](#216-straight-through-estimator-ste)
   - [GPTQ](#217-gptq)
   - [AWQ](#218-awq)
   - [GGUF](#219-gguf)
   - [Q4_K_M](#220-q4_k_m)
   - [Quantization vs Compression](#221-quantization-vs-normal-compression)
   - [Memory Calculations](#222-memory-calculations)
   - [Quality Evaluation](#223-quality-evaluation)
   - [Quantization Summary](#224-quantization-summary)
3. [Part II — Attention Variants](#3-part-ii--attention-variants)
   - [Self-Attention](#31-self-attention)
   - [Why Full Attention is Expensive](#32-why-full-attention-is-expensive)
   - [Causal Attention](#33-causal-attention)
   - [Sliding Window Attention](#34-sliding-window-attention-swa)
   - [SWA Complexity](#35-swa-complexity)
   - [SWA and Long-Range Information](#36-swa-and-long-range-information)
   - [Sparse/Block Attention](#37-sparseblock-attention)
   - [Longformer and BigBird Patterns](#38-longformer-and-bigbird-patterns)
   - [Differential Attention](#39-differential-attention)
   - [Attention Sinks](#310-attention-sinks)
   - [MHA](#311-multi-head-attention-mha)
   - [MQA](#312-multi-query-attention-mqa)
   - [GQA](#313-grouped-query-attention-gqa)
   - [KV Cache](#314-kv-cache)
   - [Why GQA Saves Memory](#315-why-gqa-saves-memory)
   - [FlashAttention](#316-flashattention)
   - [Important Distinctions](#317-important-distinctions)
   - [Attention Variants Summary](#318-attention-variants-summary)
4. [Part III — Scaling Laws](#4-part-iii--scaling-laws)
   - [What Scaling Laws Ask](#41-what-scaling-laws-ask)
   - [Parameters and Tokens](#42-parameters-and-tokens)
   - [Training Compute](#43-training-compute)
   - [The 6ND Formula](#44-the-6nd-formula)
   - [Bigger Model vs More Data](#45-bigger-model-vs-more-data)
   - [GPT-3 Example](#46-gpt-3-example)
   - [Chinchilla](#47-chinchilla)
   - [Tokens per Parameter](#48-tokens-per-parameter)
   - [Hoffmann Scaling Law](#49-hoffmann-scaling-law)
   - [Understanding Each Term](#410-understanding-each-term)
   - [Diminishing Returns](#411-diminishing-returns)
   - [Compute-Optimal Training](#412-compute-optimal-training)
   - [Derivation Under Fixed Compute](#413-derivation-under-fixed-compute)
   - [Why N and D Scale as sqrt(C)](#414-why-n-and-d-scale-as-sqrtc)
   - [Under-Training](#415-under-training)
   - [Over-Training](#416-over-training)
   - [Training-Optimal vs Inference-Optimal](#417-training-optimal-vs-inference-optimal)
   - [Llama-Style High Token/Parameter Ratios](#418-high-tokenparameter-ratios)
   - [Emergence](#419-emergence)
   - [Data Quality](#420-data-quality)
   - [Mixture-of-Experts](#421-mixture-of-experts-moe)
   - [Post-Training](#422-post-training)
   - [Multimodality](#423-multimodality)
   - [Synthetic Data](#424-synthetic-data)
   - [Optimizers and Effective Compute](#425-optimizers-and-effective-compute)
   - [Scaling Laws Visualizer Example](#426-scaling-laws-visualizer-example)
   - [Scaling Laws Summary](#427-scaling-laws-summary)
5. [Part IV — Connecting the Three Topics](#5-part-iv--connecting-the-three-topics)
6. [Formula Sheet](#6-formula-sheet)
7. [Concept Comparison Tables](#7-concept-comparison-tables)
8. [Common Confusions](#8-common-confusions)
9. [Mental Models](#9-mental-models)
10. [Final Revision Checklist](#10-final-revision-checklist)

---

# 1. Big Picture

These three topics solve **different problems** in modern LLMs.

## 1.1 Quantization

Quantization asks:

> **How can I represent the numbers inside a trained model using fewer bits?**

For example:

```text
FP32 → FP16 → INT8 → INT4
```

Fewer bits generally mean:

- less memory
- lower bandwidth requirements
- potentially faster inference
- ability to run larger models on smaller hardware

But fewer bits also mean:

- less numerical precision
- quantization error
- possible quality degradation

---

## 1.2 Attention Variants

Attention asks:

> **How can the Transformer process context more efficiently, especially when the sequence is very long?**

Standard self-attention has:

\[
O(N^2)
\]

attention interactions for sequence length \(N\).

Attention variants try to reduce memory/compute or improve behavior.

Examples:

- Sliding Window Attention
- Sparse Attention
- Differential Attention
- MQA
- GQA
- FlashAttention

These techniques do **not all solve the same problem**.

---

## 1.3 Scaling Laws

Scaling laws ask:

> **Given a limited amount of compute, how should I choose model size and training data?**

The two main quantities are:

- \(N\): number of model parameters
- \(D\): number of training tokens

A simplified training-compute relationship is:

\[
C \approx 6ND
\]

where \(C\) is training FLOPs.

Scaling laws help answer questions such as:

```text
Should I train a 10B model on 1T tokens?

or

a 20B model on 500B tokens?
```

The key lesson:

> **A larger model is not automatically better if it is severely under-trained.**

---

# 2. Part I — Quantization

# 2.1 Why Quantization?

Modern LLMs contain billions of numerical values.

Consider a model with:

\[
70B = 70\times10^9
\]

parameters.

If every parameter uses FP16:

\[
2 \text{ bytes/parameter}
\]

then memory is approximately:

\[
70\times10^9\times2
\]

\[
=140\times10^9\text{ bytes}
\]

approximately:

\[
140\text{ GB}
\]

So a 70B model in FP16 needs roughly **140 GB just for the weights**.

That does not include:

- KV cache
- activations
- runtime buffers
- CUDA overhead
- framework overhead

Therefore, reducing numerical precision can dramatically reduce memory requirements.

---

# 2.2 Bits and Number Formats

A bit can represent:

\[
0 \quad \text{or} \quad 1
\]

Therefore:

\[
n\text{ bits}\Rightarrow 2^n\text{ possible bit patterns}
\]

Examples:

| Format | Bits/value | Possible patterns |
|---|---:|---:|
| FP32 | 32 | \(2^{32}\) |
| FP16 | 16 | \(2^{16}\) |
| BF16 | 16 | \(2^{16}\) |
| FP8 | 8 | \(2^8=256\) |
| INT8 | 8 | \(2^8=256\) |
| INT4 | 4 | \(2^4=16\) |

Important:

> Number of bit patterns is not the same as the number of useful decimal values in exactly the same sense.

Floating-point formats distribute their representable values non-uniformly.

---

# 2.3 Floating Point Numbers

Floating-point numbers represent a value approximately as:

\[
(-1)^S\times M\times2^E
\]

where:

- \(S\) = sign information
- \(M\) = mantissa/significand
- \(E\) = exponent

A floating-point format divides its bits into:

```text
SIGN | EXPONENT | MANTISSA
```

The exact encoding has details such as exponent bias and special values, but the most important conceptual idea is:

> **Exponent controls range. Mantissa controls precision.**

---

## 2.3.1 Exponent

More exponent bits mean a larger numerical range.

For example:

- very small values
- very large values

can both be represented.

Think:

> **Exponent = how far I can reach.**

---

## 2.3.2 Mantissa

More mantissa bits mean more precision between values.

Think:

> **Mantissa = how finely I can measure.**

So:

```text
Exponent → RANGE
Mantissa → PRECISION
```

This distinction is extremely important.

---

# 2.4 FP32, FP16, BF16, FP8

## FP32

FP32 uses:

```text
1 sign bit
8 exponent bits
23 mantissa bits
```

Total:

\[
1+8+23=32
\]

FP32 provides:

- large range
- relatively high precision
- 4 bytes/value

---

## FP16

FP16 uses:

```text
1 sign
5 exponent
10 mantissa
```

Total:

\[
1+5+10=16
\]

Memory:

\[
16/8=2\text{ bytes}
\]

Compared with FP32:

\[
4/2=2
\]

So FP16 needs roughly **half the storage**.

---

## BF16

BF16 uses:

```text
1 sign
8 exponent
7 mantissa
```

Total:

\[
1+8+7=16
\]

The important point is:

> BF16 has the same exponent width as FP32.

Therefore BF16 retains a large dynamic range while sacrificing mantissa precision.

This is one reason BF16 is very useful for deep-learning training.

---

## FP8

Common FP8 variants include:

- E4M3
- E5M2

For example:

```text
E4M3:
1 sign + 4 exponent + 3 mantissa
```

and:

```text
E5M2:
1 sign + 5 exponent + 2 mantissa
```

The trade-off is again:

```text
More exponent → more range
More mantissa → more precision
```

---

# 2.5 Integer Quantization

Integer formats do not store floating-point values directly.

For signed INT8:

\[
-128\le q\le127
\]

There are:

\[
256
\]

possible values.

For INT4:

\[
16
\]

possible values.

A common symmetric INT8 range is:

\[
[-127,127]
\]

while implementation details can use the full signed range.

---

# 2.6 What Quantization Actually Does

Suppose the original weights are:

```text
-0.91
-0.32
 0.02
 0.47
 0.83
```

Instead of storing all these floating-point values directly, we map them to a smaller set of integer values.

Conceptually:

```text
FLOAT VALUE
     ↓
 SCALE
     ↓
 INTEGER VALUE
```

Then during computation or dequantization:

```text
INTEGER VALUE
     ↓
 SCALE
     ↓
 APPROXIMATE FLOAT VALUE
```

The central idea is:

> **Quantization replaces a high-precision numerical representation with a lower-precision representation.**

It is usually **lossy**.

---

# 2.7 Symmetric Quantization

Suppose a tensor has values in:

\[
[-a,a]
\]

For INT8, we can map the largest absolute value to the maximum integer magnitude.

Define:

\[
q_{\max}=127
\]

and:

\[
s=\frac{\max(|x|)}{127}
\]

where \(s\) is the scale.

Then:

\[
q=\operatorname{round}\left(\frac{x}{s}\right)
\]

The quantized value is \(q\).

To reconstruct:

\[
\hat{x}=q\times s
\]

where:

- \(x\) = original value
- \(q\) = quantized integer
- \(s\) = scale
- \(\hat{x}\) = reconstructed approximation

---

## Example

Suppose:

\[
x_{\max}=1.0
\]

For INT8:

\[
s=\frac{1}{127}
\]

Approximately:

\[
s\approx0.00787
\]

Now suppose:

\[
x=0.5
\]

Then:

\[
q=\operatorname{round}\left(\frac{0.5}{0.00787}\right)
\]

\[
q\approx64
\]

Dequantization:

\[
\hat{x}=64(0.00787)
\]

\[
\hat{x}\approx0.504
\]

So:

```text
Original     = 0.500
Quantized    = 64
Reconstructed≈ 0.504
```

The difference is quantization error.

---

# 2.8 Asymmetric Quantization

Symmetric quantization assumes the range is centered around zero.

But suppose:

\[
x\in[-1,3]
\]

This is not centered around zero.

Using asymmetric quantization, we introduce a **zero point**.

A common conceptual mapping is:

\[
q=\operatorname{round}\left(\frac{x}{s}\right)+z
\]

where:

- \(s\) = scale
- \(z\) = zero point

A common scale calculation is:

\[
s=\frac{x_{\max}-x_{\min}}
{q_{\max}-q_{\min}}
\]

The zero point aligns real zero with an integer value.

One way to express it is:

\[
z=q_{\min}-\frac{x_{\min}}{s}
\]

followed by appropriate rounding/clamping to the integer range.

Dequantization:

\[
\hat{x}=s(q-z)
\]

---

## Symmetric vs Asymmetric

| Property | Symmetric | Asymmetric |
|---|---|---|
| Center | Around 0 | Can shift |
| Main parameter | Scale | Scale + zero point |
| Good for | Zero-centered weights | Shifted distributions |
| Simplicity | Simpler | More complicated |

---

# 2.9 Per-Tensor Quantization

The entire tensor uses one scale.

Suppose:

```text
Tensor
↓
one scale
↓
all values quantized
```

For example:

\[
s=\frac{\max(|X|)}{127}
\]

for the entire tensor \(X\).

This is simple and cheap.

But it has a weakness.

Suppose different channels have very different ranges:

```text
Channel 1: [-0.1, 0.1]
Channel 2: [-10, 10]
```

The scale needed for Channel 2 may be far too coarse for Channel 1.

---

# 2.10 Per-Channel Quantization

Instead of one scale for the whole tensor:

```text
Tensor
├── Channel 1 → scale 1
├── Channel 2 → scale 2
├── Channel 3 → scale 3
└── Channel 4 → scale 4
```

For each channel \(c\):

\[
s_c=\frac{\max(|X_c|)}{127}
\]

Then:

\[
q_c=\operatorname{round}\left(\frac{x_c}{s_c}\right)
\]

and:

\[
\hat{x}_c=q_cs_c
\]

This usually represents differently scaled channels more accurately.

---

# 2.11 Quantization Error

Quantization cannot represent every original value.

Therefore:

\[
\hat{x}\neq x
\]

The quantization error is:

\[
e=x-\hat{x}
\]

For many values, we may measure mean squared error:

\[
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(x_i-\hat{x}_i)^2
\]

Smaller MSE means the numerical approximation is closer in this metric.

However:

> **Small numerical MSE does not automatically mean small model-quality loss.**

A tiny error in an important weight can matter more than a larger error in an unimportant weight.

This motivates methods such as GPTQ and AWQ.

---

# 2.12 Weights vs Activations vs KV Cache

Different parts of an LLM have different sensitivity to quantization.

A useful conceptual hierarchy is:

```text
Weights
Activations
KV Cache
Attention logits
```

But exact sensitivity depends heavily on the model, layer, calibration method, and quantization scheme.

---

## Weights

Weights are the learned parameters.

They are commonly quantized because:

- there are billions of them
- they consume huge amounts of memory
- they can often tolerate aggressive quantization

---

## Activations

Activations are intermediate values produced during computation.

They can be harder to quantize because their distributions can vary depending on:

- input
- layer
- token
- context

---

## KV Cache

During autoregressive generation, previous keys and values are cached.

The KV cache can become very large for long contexts and large batches.

Therefore KV-cache quantization can also save significant memory.

---

## Attention Logits

Attention logits are values before softmax:

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

Small numerical changes here can affect softmax probabilities.

Therefore aggressive quantization of sensitive attention computations can cause quality problems.

---

# 2.13 Post-Training Quantization (PTQ)

PTQ means:

> Train the model normally first, then quantize it.

Pipeline:

```text
Training
   ↓
FP16/FP32 model
   ↓
Calibration / analysis
   ↓
Quantization
   ↓
INT8 / INT4 model
```

Advantages:

- no full retraining required
- relatively convenient
- useful for deployment

Disadvantage:

- quality can degrade if quantization is too aggressive

---

# 2.14 Quantization-Aware Training (QAT)

QAT incorporates quantization behavior during training.

Conceptually:

```text
Training
  ↓
simulate quantization
  ↓
model adapts
  ↓
final quantized model
```

The model learns to tolerate quantization error.

Usually:

\[
\text{QAT quality} \geq \text{naive PTQ quality}
\]

is a useful intuition, but the actual result depends on implementation and training setup.

QAT costs additional training time.

---

# 2.15 Fake Quantization

During QAT, we often cannot directly backpropagate through integer rounding.

Instead, we simulate quantization.

Conceptually:

\[
x
\rightarrow
\text{quantize}
\rightarrow
\text{dequantize}
\rightarrow
\text{continue computation}
\]

For example:

\[
\hat{x}
=
s\cdot
\operatorname{round}(x/s)
\]

The tensor may still be represented in floating point during training, but it behaves approximately like a quantized value.

This is called **fake quantization**.

---

# 2.16 Straight-Through Estimator (STE)

Rounding is not normally differentiable in a useful way.

For:

\[
q=\operatorname{round}(x)
\]

the derivative is problematic.

QAT often uses the Straight-Through Estimator.

The forward pass behaves like quantization:

\[
y=\operatorname{round}(x)
\]

but the backward pass approximates:

\[
\frac{\partial y}{\partial x}\approx1
\]

over an appropriate range.

So the model can still receive useful gradients.

Think:

> **Forward: pretend quantization happened.**
>
> **Backward: pretend the quantization operation is approximately transparent to gradients.**

---

# 2.17 GPTQ

GPTQ is a post-training weight quantization method designed especially for large language models.

The central idea is:

> **Not all weight errors are equally harmful.**

GPTQ uses information related to the model's sensitivity to weight perturbations.

A simplified second-order view of loss change is:

\[
\Delta L
\approx
g^T\Delta w
+
\frac12
\Delta w^T H\Delta w
\]

where:

- \(g\) = gradient
- \(H\) = Hessian
- \(\Delta w\) = weight perturbation

Near a local optimum:

\[
g\approx0
\]

so:

\[
\Delta L
\approx
\frac12\Delta w^TH\Delta w
\]

This tells us that the Hessian can describe how sensitive the loss is to changes in different directions.

GPTQ uses approximate second-order information to make quantization decisions that minimize the effect on model behavior.

---

# 2.18 AWQ

AWQ stands for **Activation-aware Weight Quantization**.

Its central idea:

> Some weights are more important because of how activations interact with them.

Instead of treating every weight as equally important, AWQ identifies salient weight channels/groups using activation information.

The intuition:

```text
Activation statistics
       ↓
identify important weights/channels
       ↓
protect them
       ↓
quantize remaining weights aggressively
```

This can preserve model quality at low bit widths.

---

# 2.19 GGUF

GGUF is a model file format commonly used in local LLM ecosystems.

It can store:

- model tensors
- metadata
- tokenizer-related information
- quantized weights
- other model configuration information

Important:

> **GGUF is primarily a file/container format, not a quantization algorithm.**

For example:

```text
GPTQ → quantization method

AWQ → quantization method

GGUF → model file format
```

---

# 2.20 Q4_K_M

You may encounter names such as:

```text
Q4_K_M
```

This refers to a particular quantized representation used in GGUF/llama.cpp-style ecosystems.

The important intuition:

```text
Q4 → roughly 4-bit weight quantization
K → a family of grouped/block quantization schemes
M → a particular mixed/medium configuration
```

The exact internal implementation is more detailed than simply saying "every number is exactly four bits"; metadata, scales, grouping, and mixed precision can be involved.

Therefore:

> Do not interpret Q4_K_M as merely "the whole model is a raw array of 4-bit integers."

---

# 2.21 Quantization vs Normal Compression

These are different ideas.

## Normal Lossless Compression

Example:

```text
Original file
     ↓
ZIP
     ↓
smaller file
```

After decompression:

\[
\text{original}=\text{recovered exactly}
\]

No information is intentionally lost.

---

## Quantization

Example:

```text
0.493721
↓
INT4 representation
↓
approximately 0.5
```

The original value is not recovered exactly.

Therefore:

> **Quantization is generally lossy numerical compression.**

---

# 2.22 Memory Calculations

A simple model-weight memory formula is:

\[
\text{Memory}
\approx
\text{Number of Parameters}
\times
\text{Bytes per Parameter}
\]

Bytes per parameter:

| Precision | Approx bytes/parameter |
|---|---:|
| FP32 | 4 |
| FP16 | 2 |
| BF16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

---

## Example: 70B

FP32:

\[
70B\times4=280GB
\]

FP16:

\[
70B\times2=140GB
\]

INT8:

\[
70B\times1=70GB
\]

INT4:

\[
70B\times0.5=35GB
\]

These are approximate raw weight-memory figures.

Actual runtime memory is larger.

---

# 2.23 Why Are Many Weights Near Zero?

You may hear statements such as:

> "A large percentage of LLM weights are between -0.1 and +0.1."

The general intuition is that trained neural-network weights are often concentrated around zero rather than being uniformly distributed over a huge interval.

Training dynamics, initialization, regularization, normalization structures, and learned representations all contribute to weight distributions.

But an exact claim such as:

\[
95\%\text{ of weights}\in[-0.1,0.1]
\]

is **model- and layer-dependent**, not a universal law.

---

## Why This Matters for Quantization

Suppose almost all useful values are around:

\[
[-0.1,0.1]
\]

but one outlier is:

\[
10
\]

If a single scale is chosen based on \(10\), the quantization grid can become too coarse for the small values.

This is one reason:

- per-channel/group quantization
- outlier handling
- activation-aware methods

are useful.

---

# 2.24 Quality Evaluation

A quantized model should not be judged only by file size.

Useful measurements include:

## Perplexity

For a language model:

\[
PPL=e^{L}
\]

where \(L\) is average negative log-likelihood under the relevant convention.

Lower perplexity generally means better next-token prediction on the same evaluation setup.

---

## Task Benchmarks

Depending on the model:

- MMLU
- GSM8K
- HumanEval
- other domain-specific benchmarks

can be used.

---

## Output Comparison

Compare:

```text
FP16 model output
vs
INT8/INT4 model output
```

for the same prompts.

---

## System Metrics

Also measure:

- latency
- throughput
- memory usage
- tokens/second
- batch-size behavior

A quantization method is useful only if it provides an acceptable quality/performance trade-off.

---

# 2.25 Quantization Summary

Remember:

```text
Quantization
     ↓
fewer bits per numerical value
     ↓
less memory + bandwidth
     ↓
potentially cheaper/faster inference
     ↓
but introduces approximation error
```

Core formulas:

\[
s=\frac{x_{\max}}{q_{\max}}
\]

for a simple symmetric positive-range illustration, or more generally:

\[
s=\frac{\max(|x|)}{q_{\max}}
\]

for symmetric signed quantization.

Then:

\[
q=\operatorname{round}(x/s)
\]

and:

\[
\hat{x}=qs
\]

For asymmetric quantization:

\[
s=
\frac{x_{\max}-x_{\min}}
{q_{\max}-q_{\min}}
\]

\[
z=q_{\min}-\frac{x_{\min}}{s}
\]

\[
\hat{x}=s(q-z)
\]

---

# 3. Part II — Attention Variants

# 3.1 Self-Attention

For an input sequence:

\[
X\in\mathbb{R}^{N\times d}
\]

we create:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

The attention score matrix is:

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

Then:

\[
A=\operatorname{softmax}(S)
\]

Finally:

\[
O=AV
\]

So the standard attention equation is:

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
\]

---

# 3.2 Why Full Attention is Expensive

Suppose:

\[
N=8192
\]

tokens.

The attention matrix has shape:

\[
8192\times8192
\]

Number of entries:

\[
8192^2=67,108,864
\]

That is more than 67 million attention scores.

In general:

\[
N\times N
\]

means:

\[
O(N^2)
\]

interactions.

If sequence length doubles:

\[
N\rightarrow2N
\]

then:

\[
N^2\rightarrow(2N)^2=4N^2
\]

So:

> **Doubling sequence length can quadruple the attention interaction count.**

This is the central motivation behind efficient attention variants.

---

# 3.3 Causal Attention

In autoregressive language modeling, token \(i\) should not see future token \(j>i\).

Therefore:

\[
A_{ij}=0
\]

for:

\[
j>i
\]

A common implementation adds a mask:

\[
S_{ij}=-\infty
\]

for forbidden positions.

Then softmax gives:

\[
\operatorname{softmax}(-\infty)\approx0
\]

So:

```text
Token 1 → sees token 1
Token 2 → sees 1,2
Token 3 → sees 1,2,3
...
```

---

# 3.4 Sliding Window Attention (SWA)

Instead of allowing token \(i\) to attend to all previous tokens, let it attend only to a fixed window.

For causal SWA with window size \(W\):

\[
j\in[i-W+1,i]
\]

subject to:

\[
j\ge0
\]

Example:

\[
W=4
\]

Token 10 can attend approximately to:

```text
7, 8, 9, 10
```

rather than:

```text
0, 1, 2, ..., 10
```

---

# 3.5 SWA Complexity

Full causal attention still has approximately:

\[
O(N^2)
\]

interactions.

With a fixed window \(W\):

\[
O(NW)
\]

interactions.

If:

\[
N=8192
\]

and:

\[
W=1024
\]

then the rough interaction ratio is:

\[
\frac{N^2}{NW}
=
\frac{N}{W}
\]

\[
=
\frac{8192}{1024}
=
8
\]

So the local attention pattern has roughly **8 times fewer token interactions** in this simplified comparison.

---

# 3.6 SWA and Long-Range Information

A major concern:

> If every layer only sees a local window, how can information travel from a very old token to the current token?

Suppose:

\[
W=1024
\]

A token cannot directly access a token thousands of positions away.

But information can propagate across layers.

Conceptually:

```text
Layer 1:
A sees nearby B

Layer 2:
information from B can move farther

Layer 3:
it moves farther again

...
```

With enough layers, the effective receptive field can grow.

A rough intuition is:

\[
\text{effective reach}\sim L\times W
\]

for \(L\) layers, depending on the exact architecture and overlap.

This is why architectures may:

- interleave global attention layers
- use larger windows
- combine local and global mechanisms

---

# 3.7 Sparse/Block Attention

Instead of computing every:

\[
(i,j)
\]

pair, sparse attention computes only selected pairs.

A full attention matrix looks like:

```text
████████████████
████████████████
████████████████
████████████████
```

Sparse attention looks conceptually like:

```text
██░░██░░░░░░██
░███░░██░░░░░
██░░██░░██░░░
...
```

The important idea:

> **Most possible query-key pairs are never computed.**

---

# 3.8 Longformer and BigBird Patterns

Different sparse-attention designs use different connectivity patterns.

## Local Attention

Each token attends to nearby tokens.

Good for:

- local syntax
- nearby context
- efficient processing

---

## Global Attention

Selected tokens can attend broadly.

Useful for:

- important summary tokens
- special positions
- long-range information

---

## Random Attention

Some connections are selected randomly.

The goal is to improve connectivity without creating a dense \(N\times N\) matrix.

---

## Strided Attention

A token attends to positions separated by a fixed stride.

For example:

```text
0, 4, 8, 12, 16, ...
```

This can provide long-range access.

---

# 3.9 Sparse Complexity

The exact complexity depends on the sparsity pattern.

For some structured sparse schemes:

\[
O(N\sqrt{N})
\]

can be achieved.

But an important engineering point is:

> Setting unwanted attention entries to zero is not automatically enough.

If the implementation still computes the entire dense matrix and then masks it, you may still pay most of the dense computational cost.

True efficiency requires the hardware/kernel implementation to **skip the unnecessary computation**.

---

# 3.10 Differential Attention

Differential Attention modifies the attention mechanism by subtracting two attention patterns.

First:

\[
A_1=
\operatorname{softmax}
\left(
\frac{Q_1K_1^T}{\sqrt{d}}
\right)
\]

Second:

\[
A_2=
\operatorname{softmax}
\left(
\frac{Q_2K_2^T}{\sqrt{d}}
\right)
\]

Then:

\[
\operatorname{DiffAttn}
=
(A_1-\lambda A_2)V
\]

where:

\[
\lambda
\]

controls the amount of subtraction.

---

## Intuition

Imagine:

```text
A1 = useful attention + unwanted attention

A2 = another estimate of unwanted/common attention
```

Then:

\[
A_1-\lambda A_2
\]

can suppress some common/unwanted attention patterns.

The goal is not primarily to reduce the \(N^2\) computation.

It changes the **attention behavior**.

---

# 3.11 Differential Attention Complexity

We calculate two attention patterns:

\[
A_1
\]

and:

\[
A_2
\]

Therefore, compared with ordinary attention, there is roughly an additional attention computation.

Conceptually:

\[
O(2N^2)
\]

rather than:

\[
O(N^2)
\]

for the corresponding attention-score computation.

So:

> **Differential Attention is not primarily a compute-saving technique.**

Its purpose is to modify/select attention patterns.

---

# 3.12 Attention Sinks

An **attention sink** is a token/position that can receive a surprisingly large amount of attention even when its semantic content is not obviously important.

In long-context generation, attention may concentrate on special early positions.

This can create problems when aggressively truncating or changing the context structure.

Differential-attention-style mechanisms can be motivated partly by reducing unwanted/common attention components.

Important:

> "Attention sink" is a behavioral phenomenon, not simply "the first token is always important."

---

# 3.13 Multi-Head Attention (MHA)

Standard Multi-Head Attention has multiple independent query, key, and value heads.

Suppose there are:

\[
H
\]

heads.

Each head has:

\[
Q_h,K_h,V_h
\]

and computes:

\[
O_h=
\operatorname{softmax}
\left(
\frac{Q_hK_h^T}{\sqrt{d_h}}
\right)V_h
\]

Then the outputs are concatenated and projected.

---

## Why Multiple Heads?

Different heads can learn different relationships.

Conceptually:

```text
Head 1 → syntax
Head 2 → long-range relation
Head 3 → local relation
Head 4 → positional pattern
...
```

This is an intuition, not a guarantee that every head has one clean human-interpretable role.

---

# 3.14 Multi-Query Attention (MQA)

MQA keeps many query heads but shares a single key/value head.

MHA:

```text
Q1 K1 V1
Q2 K2 V2
Q3 K3 V3
Q4 K4 V4
```

MQA:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤ → shared K,V
Q4 ─┘
```

So:

- many Q heads
- one K head
- one V head

This greatly reduces KV-cache memory.

---

# 3.15 Grouped-Query Attention (GQA)

GQA sits between MHA and MQA.

Suppose:

\[
H_Q=16
\]

query heads.

Instead of:

\[
H_{KV}=16
\]

we might use:

\[
H_{KV}=4
\]

Then query heads are divided into groups.

For example:

```text
Q1 Q2 Q3 Q4   → K1 V1
Q5 Q6 Q7 Q8   → K2 V2
Q9 Q10 Q11 Q12 → K3 V3
Q13 Q14 Q15 Q16 → K4 V4
```

So:

\[
H_Q=16
\]

but:

\[
H_{KV}=4
\]

---

# 3.16 KV Cache

During autoregressive generation, we repeatedly predict one token at a time.

For previous tokens, their keys and values can be reused.

Instead of recomputing them every generation step, we store:

```text
K cache
V cache
```

This is the **KV cache**.

---

## Why KV Cache Becomes Huge

A simplified KV-cache memory relationship is:

\[
M_{KV}
\propto
L\times N\times H_{KV}\times d_h\times2
\]

where:

- \(L\) = number of layers
- \(N\) = cached sequence length
- \(H_{KV}\) = number of KV heads
- \(d_h\) = head dimension
- \(2\) = K and V

If using \(b\) bytes per element:

\[
M_{KV}
\approx
2LN H_{KV}d_hb
\]

The factor 2 comes from storing both K and V.

---

# 3.17 Why GQA Saves Memory

Compare:

\[
H_Q=16
\]

MHA:

\[
H_{KV}=16
\]

GQA:

\[
H_{KV}=4
\]

Then:

\[
\frac{16}{4}=4
\]

So the KV cache can be approximately:

\[
4\times
\]

smaller, all else equal.

General reduction factor:

\[
\boxed{
\frac{H_Q}{H_{KV}}
}
\]

---

## Important Distinction

GQA does **not** primarily reduce the number of token-token attention pairs.

It mainly reduces:

> **the amount of K/V data that must be stored and moved.**

This is different from SWA.

---

# 3.18 FlashAttention

FlashAttention is an efficient implementation of attention.

A common misunderstanding is:

> "FlashAttention changes full attention from \(O(N^2)\) to \(O(N)\)."

That is not the correct basic idea.

The attention computation still has the same dense interaction structure.

The major improvement comes from reducing expensive memory traffic and avoiding materializing the full attention matrix in the usual way.

Conceptually:

```text
Naive:

QKᵀ
 ↓
store huge attention matrix
 ↓
softmax
 ↓
multiply by V

FlashAttention:

process attention in blocks
 ↓
keep useful values in fast memory
 ↓
avoid storing full attention matrix
 ↓
reduce memory movement
```

Therefore:

> **FlashAttention improves implementation efficiency and memory usage; it does not magically remove the \(N^2\) dense attention interactions.**

---

# 3.19 Important Distinctions

This table is extremely important.

| Technique | Main purpose |
|---|---|
| Full Attention | Maximum dense connectivity |
| SWA | Restrict attention to local window |
| Sparse Attention | Compute selected attention pairs |
| Differential Attention | Modify/suppress attention patterns |
| MHA | Multiple independent K/V heads |
| MQA | Share one K/V head |
| GQA | Share K/V across groups |
| FlashAttention | Efficient implementation of attention |

---

# 3.20 Attention Variants Summary

Think of them by the problem they solve.

## Want fewer attention interactions?

Use:

```text
SWA
Sparse Attention
```

---

## Want smaller KV cache?

Use:

```text
MQA
GQA
```

---

## Want different attention behavior?

Use:

```text
Differential Attention
```

---

## Want better memory traffic/kernel efficiency?

Use:

```text
FlashAttention
```

---

# 4. Part III — Scaling Laws

# 4.1 What Scaling Laws Ask

Imagine you have a fixed compute budget.

You can spend it on:

```text
bigger model
```

or:

```text
more training data
```

The question is:

> **How should compute be divided between model size and data?**

This is what scaling laws study.

---

# 4.2 Parameters and Tokens

Let:

\[
N=\text{number of model parameters}
\]

and:

\[
D=\text{number of training tokens}
\]

Example:

```text
Model = 70B parameters

Training data = 1.4T tokens
```

Then:

\[
N=70\times10^9
\]

\[
D=1.4\times10^{12}
\]

---

# 4.3 Training Compute

A common approximation for dense Transformer training is:

\[
\boxed{
C\approx6ND
}
\]

where:

- \(C\) = training FLOPs
- \(N\) = parameters
- \(D\) = training tokens

The constant 6 is an approximation based on forward and backward computation for dense models.

It is not a universal physical constant.

Actual compute depends on:

- architecture
- sequence length
- attention implementation
- sparsity
- MoE
- optimizer
- training details

---

# 4.4 The 6ND Formula

Suppose:

\[
N=1B
\]

and:

\[
D=20B
\]

Then:

\[
C\approx6(1B)(20B)
\]

\[
=120\times10^{18}
\]

\[
=1.2\times10^{20}
\]

FLOPs.

---

# 4.5 Bigger Model vs More Data

Suppose you have a fixed compute budget:

\[
C=6ND
\]

Then:

\[
D=\frac{C}{6N}
\]

This means:

> If you make \(N\) larger while keeping compute fixed, you must reduce \(D\).

So:

```text
Bigger model
      ↕
Less data
```

and:

```text
Smaller model
      ↕
More data
```

The question is which balance gives the lowest final loss.

---

# 4.6 GPT-3 Example

GPT-3 had approximately:

\[
175B
\]

parameters and around:

\[
300B
\]

training tokens under commonly cited estimates.

The ratio was approximately:

\[
\frac{300B}{175B}
\approx1.7
\]

tokens per parameter.

This is much smaller than the later Chinchilla compute-optimal ratio.

The important lesson is:

> A very large model can be trained with too little data relative to its capacity.

---

# 4.7 Chinchilla

Hoffmann et al. studied compute-optimal language-model scaling.

The major conclusion was approximately:

> For a fixed training-compute budget, model size and training tokens should both increase rather than putting almost all compute into model parameters.

A widely used simplified rule from the Chinchilla analysis is:

\[
\boxed{
D\approx20N
}
\]

where:

- \(D\) = training tokens
- \(N\) = parameters

So approximately:

\[
20\text{ tokens per parameter}
\]

under the original compute-optimal setting.

---

## Example: 70B

\[
D\approx20(70B)
\]

\[
D\approx1.4T
\]

tokens.

This illustrates why a 70B model trained on roughly 1.4T tokens can be viewed as approximately compute-optimal under that scaling rule.

---

# 4.8 Tokens per Parameter

Define:

\[
r=\frac{D}{N}
\]

Then:

- small \(r\) → relatively little training data
- large \(r\) → relatively more training data

Chinchilla-style compute-optimal training suggested:

\[
r\approx20
\]

for the relevant regime.

But this is **not a universal law for every modern training objective**.

---

# 4.9 Hoffmann Scaling Law

A commonly presented form is:

\[
\boxed{
L(N,D)
=
\frac{A}{N^\alpha}
+
\frac{B}{D^\beta}
+
E
}
\]

where:

- \(L\) = training loss / validation loss proxy
- \(N\) = model parameters
- \(D\) = training tokens
- \(A,B\) = fitted constants
- \(\alpha,\beta\) = scaling exponents
- \(E\) = approximate irreducible/fitted floor

The lesson values often presented are approximately:

\[
\alpha\approx0.34
\]

\[
\beta\approx0.28
\]

with fitted constants such as:

\[
A\approx406
\]

\[
B\approx411
\]

\[
E\approx1.69
\]

These fitted constants depend on the particular scaling-law formulation and experimental setup.

---

# 4.10 Understanding Each Term

The model-related term is:

\[
\frac{A}{N^\alpha}
\]

As \(N\) increases:

\[
N^\alpha\uparrow
\]

therefore:

\[
\frac{A}{N^\alpha}\downarrow
\]

So:

> Bigger model → lower model-capacity contribution to loss.

---

The data-related term is:

\[
\frac{B}{D^\beta}
\]

As \(D\) increases:

\[
D^\beta\uparrow
\]

therefore:

\[
\frac{B}{D^\beta}\downarrow
\]

So:

> More training data → lower data-related contribution to loss.

---

The floor:

\[
E
\]

does not disappear merely by making \(N\) and \(D\) larger within this simple fitted model.

It represents an asymptotic/fitted baseline term.

---

# 4.11 Diminishing Returns

This is one of the most important scaling-law ideas.

Suppose:

\[
L_N\propto N^{-0.34}
\]

If we double \(N\):

\[
(2N)^{-0.34}
=
2^{-0.34}N^{-0.34}
\]

Since:

\[
2^{0.34}\approx1.27
\]

the model-dependent term becomes roughly:

\[
\frac{1}{1.27}
\]

of its old value.

Similarly:

\[
D^{-0.28}
\]

If we double \(D\):

\[
2^{0.28}\approx1.21
\]

So the data-dependent term becomes roughly:

\[
\frac{1}{1.21}
\]

of its old value.

Therefore:

> Doubling resources gives improvement, but not a doubling of quality.

This is **diminishing returns**.

---

# 4.12 Compute-Optimal Training

We have:

\[
C=6ND
\]

and:

\[
L(N,D)
=
\frac{A}{N^\alpha}
+
\frac{B}{D^\beta}
+
E
\]

For fixed \(C\), we want to choose \(N,D\) that minimize \(L\).

From:

\[
C=6ND
\]

we get:

\[
D=\frac{C}{6N}
\]

Substitute into the loss:

\[
L(N)
=
\frac{A}{N^\alpha}
+
\frac{B}
{
\left(\frac{C}{6N}\right)^\beta
}
+
E
\]

Simplify:

\[
L(N)
=
A N^{-\alpha}
+
B
\left(\frac{6N}{C}\right)^\beta
+
E
\]

Then differentiate with respect to \(N\):

\[
\frac{dL}{dN}
=
-\alpha A N^{-\alpha-1}
+
\beta B
\left(\frac{6}{C}\right)^\beta
N^{\beta-1}
\]

Set:

\[
\frac{dL}{dN}=0
\]

Therefore:

\[
\alpha A N^{-\alpha-1}
=
\beta B
\left(\frac{6}{C}\right)^\beta
N^{\beta-1}
\]

Rearranging gives the optimal scaling relationship.

The exact exponent depends on \(\alpha,\beta\).

---

# 4.13 Deriving the Optimal Relationship

Starting from:

\[
\alpha A N^{-\alpha-1}
=
\beta B
\left(\frac{6}{C}\right)^\beta
N^{\beta-1}
\]

Move the powers of \(N\):

\[
N^{\alpha+\beta}
=
\frac{\alpha A}{\beta B}
\left(\frac{C}{6}\right)^\beta
\]

Therefore:

\[
\boxed{
N_{\text{opt}}
=
\left[
\frac{\alpha A}{\beta B}
\right]^{\frac{1}{\alpha+\beta}}
\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}
}
\]

Then:

\[
D_{\text{opt}}
=
\frac{C}{6N_{\text{opt}}}
\]

So:

\[
\boxed{
D_{\text{opt}}
\propto
C^{\frac{\alpha}{\alpha+\beta}}
}
\]

and:

\[
\boxed{
N_{\text{opt}}
\propto
C^{\frac{\beta}{\alpha+\beta}}
}
\]

This is the general scaling-law result from the simple power-law formulation.

---

# 4.14 Why N and D Scale as sqrt(C)

The original Chinchilla compute-optimal analysis gives an approximately linear relationship:

\[
D\propto N
\]

If:

\[
D=kN
\]

then:

\[
C=6ND
\]

becomes:

\[
C=6N(kN)
\]

\[
C=6kN^2
\]

Therefore:

\[
N^2=\frac{C}{6k}
\]

and:

\[
N\propto\sqrt C
\]

Since:

\[
D=kN
\]

we also have:

\[
D\propto\sqrt C
\]

This explains the familiar idea:

\[
\boxed{
N_{\text{opt}}\propto\sqrt C
}
\]

\[
\boxed{
D_{\text{opt}}\propto\sqrt C
}
\]

---

# 4.15 The ~20 Tokens/Parameter Rule

Suppose:

\[
D\approx20N
\]

Then:

\[
C=6N(20N)
\]

\[
C=120N^2
\]

Therefore:

\[
N\approx\sqrt{\frac{C}{120}}
\]

and:

\[
D\approx20\sqrt{\frac{C}{120}}
\]

The exact constants depend on the empirical scaling-law formulation.

The key conceptual point is:

> Under this compute-optimal regime, increasing compute should increase both model size and training data.

---

# 4.16 Under-Training

A model is **under-trained** when its capacity is large relative to the amount of data it receives.

Example:

```text
Huge model
+
not enough training tokens
```

The model has capacity that has not been fully exploited.

Think of a student with:

```text
excellent brain
+
only reads 10 pages of a textbook
```

The student has capacity but insufficient exposure to the material.

---

# 4.17 Over-Training

"Over-training" in scaling-law discussions can be confusing.

It does not necessarily mean:

> "The model is overfitting and therefore bad."

It can mean:

> Training a model on substantially more tokens than the original compute-optimal training ratio would suggest.

For example:

```text
Compute-optimal for training:
70B → ~1.4T tokens

Inference-oriented strategy:
70B → many more tokens
```

Why might this be useful?

Because training compute is paid mostly once, while inference compute may be paid:

```text
millions/billions of times
```

---

# 4.18 Training-Optimal vs Inference-Optimal

This distinction is extremely important.

## Training-Compute-Optimal

Question:

> Given a fixed training FLOP budget, what model/data combination minimizes loss?

This is the setting associated with Chinchilla-style compute-optimal scaling.

---

## Inference-Aware

Question:

> What model should I deploy if inference cost matters repeatedly?

A smaller model trained longer can sometimes be attractive because:

```text
Training cost
    ↓
paid once

Inference cost
    ↓
paid repeatedly
```

Therefore the best deployment choice need not be the same as the training-compute-optimal choice.

---

# 4.19 High Token/Parameter Ratios

Suppose an 8B model is trained on:

\[
15T
\]

tokens.

Then:

\[
\frac{15T}{8B}
=
1875
\]

tokens per parameter.

That is vastly larger than:

\[
20
\]

tokens/parameter.

This is an example of what is sometimes called **over-training relative to the original Chinchilla compute-optimal ratio**.

The word "over-training" here refers to the token/parameter ratio, not necessarily bad training.

---

# 4.20 Emergence

You may hear:

> "A capability suddenly emerges at a certain model size."

This can be misleading.

Suppose a model's true underlying capability improves smoothly:

```text
0.60
0.65
0.70
0.75
0.80
```

But evaluation uses an exact-match threshold.

Then measured performance may look like:

```text
20%
20%
35%
70%
90%
```

This can create an appearance of sudden emergence.

Therefore:

> Some apparent emergence can be caused by the interaction between a smooth underlying capability and a discontinuous evaluation metric.

This does not mean every reported emergent capability is purely a measurement artifact. The measurement design and capability itself both matter.

---

# 4.21 Data Quality

Scaling laws often count:

\[
D=\text{number of tokens}
\]

But tokens are not equally valuable.

Consider:

```text
100B high-quality tokens
```

versus:

```text
100B noisy/duplicated/low-information tokens
```

They should not necessarily be expected to provide identical learning value.

Therefore:

> Raw token count is an imperfect measure of useful training information.

Better data can effectively improve the value of a given compute budget.

---

# 4.22 Mixture-of-Experts (MoE)

MoE models complicate simple parameter-count scaling.

Suppose:

\[
N_{\text{total}}=100B
\]

but only:

\[
N_{\text{active}}=20B
\]

parameters are activated for a particular token.

Then:

```text
Total parameters = 100B
Active parameters/token ≈ 20B
```

This means:

- storage depends heavily on total parameters
- computation per token depends more on active parameters
- training/inference scaling cannot be described by total parameter count alone

Therefore:

> For MoE, distinguish **total parameters** from **active parameters/FLOPs**.

---

# 4.23 Post-Training

Scaling-law discussions often focus on pretraining.

But modern LLM performance also depends on:

- supervised fine-tuning
- preference optimization
- RL
- instruction tuning
- DPO
- other post-training methods

A model's final behavior is therefore not determined only by:

\[
N
\]

and:

\[
D_{\text{pretraining}}
\]

Post-training can significantly change:

- instruction following
- helpfulness
- formatting
- reasoning behavior
- safety behavior
- task performance

So simple pretraining scaling laws do not fully describe final deployed-model quality.

---

# 4.24 Multimodality

A text-only model mainly deals with text tokens.

A multimodal model may process:

- text tokens
- image patches/tokens
- audio tokens
- video representations

Now the simple variable:

\[
D=\text{text tokens}
\]

is insufficient to describe all training information.

We need to think about:

- modality mixture
- tokenization
- representation density
- compute per modality
- quality of data

Therefore multimodal scaling is more complicated than simply counting text tokens.

---

# 4.25 Synthetic Data

Synthetic data introduces another complication.

Suppose a model generates training examples for another model.

The number of tokens is easy to count.

But the information value is harder to measure.

Questions include:

- Is the synthetic data correct?
- Is it diverse?
- Is it repetitive?
- Does it contain reasoning traces?
- Is it derived from the same model family?
- Does it compound errors?

Therefore:

> "More synthetic tokens" does not automatically mean "more useful data."

---

# 4.26 Optimizers and Effective Compute

Different optimizers can make training more efficient.

For example, changes in optimization algorithms may allow a model to reach a target loss with less compute.

This motivates the concept of:

> **Effective compute**

Suppose Model A reaches a target loss using:

\[
C_A
\]

while Model B reaches the same target loss using:

\[
C_B
\]

If:

\[
C_B<C_A
\]

then B can be viewed as more compute-efficient for that setup.

However, the exact "compute multiplier" depends on:

- architecture
- optimizer
- dataset
- training regime
- target metric

So optimizer improvements do not imply a universal constant improvement across all models.

---

# 4.27 Scaling Laws Visualizer Example

Suppose:

\[
N=10^9
\]

parameters.

This is:

\[
1B
\]

parameters.

Suppose:

\[
D=10^{10.3}
\]

tokens.

Since:

\[
10^{10.3}\approx20B
\]

we have approximately:

\[
D\approx20B
\]

Therefore:

\[
\frac{D}{N}
\approx
\frac{20B}{1B}
=20
\]

This matches the simplified Chinchilla-style ratio.

Compute:

\[
C\approx6ND
\]

\[
C\approx6(1B)(20B)
\]

\[
C\approx1.2\times10^{20}
\]

FLOPs.

---

# 4.28 Scaling Laws Summary

Core formulas:

### Training compute

\[
\boxed{C\approx6ND}
\]

### Hoffmann-style loss law

\[
\boxed{
L(N,D)
=
\frac{A}{N^\alpha}
+
\frac{B}{D^\beta}
+
E
}
\]

### Chinchilla-style compute-optimal ratio

\[
\boxed{
D\approx20N
}
\]

### Approximate compute-optimal scaling

\[
\boxed{
N_{\text{opt}}\propto\sqrt C
}
\]

\[
\boxed{
D_{\text{opt}}\propto\sqrt C
}
\]

### Tokens per parameter

\[
\boxed{
r=\frac{D}{N}
}
\]

---

# 5. Part IV — Connecting the Three Topics

Now connect everything.

These topics occur at different stages of the LLM lifecycle.

```text
                    LLM DEVELOPMENT
                          │
                          ▼
                 ┌─────────────────┐
                 │  Scaling Laws   │
                 └─────────────────┘
                          │
                          │
              Decide model/data scale
                          │
                          ▼
                 ┌─────────────────┐
                 │   Transformer   │
                 │    Training     │
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Attention Design│
                 └─────────────────┘
                          │
              How context is processed
                          │
                          ▼
                 ┌─────────────────┐
                 │ Trained Model   │
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Quantization   │
                 └─────────────────┘
                          │
                  Compress numbers
                          │
                          ▼
                 ┌─────────────────┐
                 │ Efficient Model │
                 │    Serving      │
                 └─────────────────┘
```

---

## 5.1 Scaling Laws = How Much to Build

Scaling laws answer:

```text
How many parameters?
How many tokens?
How much training compute?
```

Main variables:

\[
N,D,C
\]

---

## 5.2 Attention Variants = How to Process Context

Attention mechanisms answer:

```text
Which tokens should interact?
How many interactions should happen?
How much KV cache is required?
How efficiently can attention be implemented?
```

Main variable:

\[
N_{\text{sequence}}
\]

---

## 5.3 Quantization = How to Represent the Numbers

Quantization answers:

```text
How many bits should each numerical value use?
```

Main variables:

\[
b=\text{bits/value}
\]

and memory:

\[
M\approx Nb/8
\]

for raw parameter storage.

---

# 5.4 One Example

Imagine a:

\[
70B
\]

parameter LLM.

### Scaling

You decide to train it on:

\[
1.4T
\]

tokens.

Then:

\[
D/N
=
1.4T/70B
=
20
\]

tokens/parameter.

---

### Attention

During inference, users send:

```text
100k-token contexts
```

Full attention becomes expensive because:

\[
N^2
\]

grows enormously.

You might therefore consider:

- local/sliding attention
- sparse/global patterns
- GQA
- FlashAttention

---

### Quantization

The FP16 weights need approximately:

\[
70B\times2
=
140GB
\]

Quantized INT4 storage is approximately:

\[
70B\times0.5
=
35GB
\]

before accounting for quantization metadata and runtime overhead.

So:

```text
Scaling
→ decide 70B + 1.4T

Attention
→ make long-context processing practical

Quantization
→ make deployment memory-efficient
```

---

# 6. Formula Sheet

## Quantization

### Symmetric scale

\[
\boxed{
s=
\frac{\max(|x|)}{q_{\max}}
}
\]

### Quantization

\[
\boxed{
q=
\operatorname{round}(x/s)
}
\]

### Dequantization

\[
\boxed{
\hat{x}=qs
}
\]

### Error

\[
\boxed{
e=x-\hat{x}
}
\]

### MSE

\[
\boxed{
MSE=
\frac1n
\sum_i(x_i-\hat{x}_i)^2
}
\]

### Asymmetric scale

\[
\boxed{
s=
\frac{x_{\max}-x_{\min}}
{q_{\max}-q_{\min}}
}
\]

### Asymmetric dequantization

\[
\boxed{
\hat{x}=s(q-z)
}
\]

---

## Attention

### Query

\[
Q=XW_Q
\]

### Key

\[
K=XW_K
\]

### Value

\[
V=XW_V
\]

### Attention scores

\[
\boxed{
S=\frac{QK^T}{\sqrt{d_k}}
}
\]

### Attention weights

\[
\boxed{
A=\operatorname{softmax}(S)
}
\]

### Output

\[
\boxed{
O=AV
}
\]

### Full attention complexity

\[
\boxed{
O(N^2)
}
\]

### Sliding window

\[
\boxed{
O(NW)
}
\]

### GQA KV reduction factor

\[
\boxed{
\frac{H_Q}{H_{KV}}
}
\]

### Approximate KV cache

\[
\boxed{
M_{KV}
\approx
2LN H_{KV}d_hb
}
\]

---

## Scaling

### Training compute

\[
\boxed{
C\approx6ND
}
\]

### Tokens per parameter

\[
\boxed{
r=\frac{D}{N}
}
\]

### Chinchilla-style ratio

\[
\boxed{
D\approx20N
}
\]

### Scaling loss

\[
\boxed{
L=
\frac{A}{N^\alpha}
+
\frac{B}{D^\beta}
+
E
}
\]

### Fixed compute

\[
\boxed{
D=\frac{C}{6N}
}
\]

### Approximate compute-optimal scaling under the linear ratio

\[
\boxed{
N_{\text{opt}}\propto\sqrt C
}
\]

\[
\boxed{
D_{\text{opt}}\propto\sqrt C
}
\]

---

# 7. Concept Comparison Tables

## 7.1 Number Formats

| Format | Bits | Main characteristic |
|---|---:|---|
| FP32 | 32 | High precision/range |
| FP16 | 16 | Half FP32 storage |
| BF16 | 16 | Large range, lower precision |
| FP8 | 8 | Very compact floating point |
| INT8 | 8 | Integer quantization |
| INT4 | 4 | Very aggressive compression |

---

## 7.2 Quantization Methods

| Method | Stage | Main idea |
|---|---|---|
| PTQ | After training | Quantize trained model |
| QAT | During training | Train with quantization effects |
| GPTQ | PTQ | Sensitivity/second-order-aware weight quantization |
| AWQ | PTQ | Activation-aware protection of important weights |
| GGUF | Deployment format | Stores model/metadata/quantized tensors |
| Q4_K_M | Quantized format/config | Grouped low-bit representation |

---

## 7.3 Attention Methods

| Method | Main idea | Main benefit |
|---|---|---|
| MHA | Separate Q/K/V heads | Rich multi-head representation |
| MQA | Shared K/V | Small KV cache |
| GQA | Grouped/shared K/V | Memory-quality trade-off |
| SWA | Local window | Fewer interactions |
| Sparse | Selected connections | Fewer interactions |
| Differential | Difference of attention maps | Alter attention behavior |
| FlashAttention | Tiled efficient implementation | Lower memory traffic |

---

## 7.4 Complexity

| Technique | Approximate attention complexity |
|---|---:|
| Full attention | \(O(N^2)\) |
| Sliding Window | \(O(NW)\) |
| Some structured sparse schemes | \(O(N\sqrt N)\) |
| FlashAttention | Same dense interaction structure; optimized implementation |

---

## 7.5 What Each Technique Optimizes

| Problem | Technique |
|---|---|
| Model weight memory | Quantization |
| Long-context attention interactions | SWA / Sparse |
| KV cache memory | MQA / GQA |
| Attention kernel memory traffic | FlashAttention |
| Attention pattern suppression/modification | Differential Attention |
| Training compute allocation | Scaling Laws |

---

# 8. Common Confusions

## Confusion 1

### "INT4 means the model is exactly 4 times smaller than FP16."

Approximately, for the raw numerical values:

\[
16/4=4
\]

so yes, roughly 4×.

But real model files contain:

- scales
- metadata
- group information
- other tensors

Therefore actual file size may not be exactly 4× smaller.

---

# Confusion 2

### "Quantization is just ZIP compression."

No.

ZIP is generally lossless:

\[
\text{decode}(\text{encode}(x))=x
\]

Quantization generally produces:

\[
\hat{x}\neq x
\]

---

# Confusion 3

### "BF16 is less precise than FP16, so it must be worse."

Not necessarily.

BF16 has fewer mantissa bits:

```text
FP16 → 10 mantissa bits
BF16 → 7 mantissa bits
```

but BF16 has:

```text
FP16 → 5 exponent bits
BF16 → 8 exponent bits
```

So BF16 has much larger dynamic range.

---

# Confusion 4

### "FlashAttention reduces O(N²) to O(N)."

No.

Dense attention still has the same basic interaction structure.

FlashAttention mainly improves:

- memory usage
- memory traffic
- kernel efficiency

---

# Confusion 5

### "GQA reduces O(N²) attention to O(N)."

No.

GQA primarily reduces:

\[
H_{KV}
\]

and therefore KV-cache memory.

---

# Confusion 6

### "SWA and GQA do the same thing."

No.

SWA:

> reduces how many historical tokens each token attends to.

GQA:

> reduces how many distinct K/V heads must be stored.

---

# Confusion 7

### "Differential Attention is a sparse-attention method."

Not primarily.

Differential Attention computes two attention patterns and subtracts them:

\[
(A_1-\lambda A_2)V
\]

Its main purpose is to alter attention behavior.

---

# Confusion 8

### "20 tokens per parameter is a universal law."

No.

It is a simplified representation of the original Chinchilla compute-optimal regime.

Modern training can intentionally use much larger token/parameter ratios because:

- inference cost matters
- data quality matters
- architecture differs
- training objectives differ
- compute budgets differ

---

# Confusion 9

### "Over-training means overfitting."

Not necessarily.

In scaling-law language, it can simply mean:

> Training a model on more tokens than the original compute-optimal ratio would suggest.

---

# Confusion 10

### "More parameters always means a better model."

No.

A larger model with insufficient training data can be under-trained.

For fixed compute:

\[
C=6ND
\]

making \(N\) larger forces \(D\) smaller.

So the balance matters.

---

# Confusion 11

### "More tokens always means better."

Not necessarily.

Data quality matters.

```text
100B excellent tokens
```

may be more useful than:

```text
100B duplicated/noisy tokens
```

---

# Confusion 12

### "GGUF is a quantization algorithm."

No.

GGUF is primarily a model file/container format.

---

# Confusion 13

### "Low-rank means a small numerical value."

No.

Rank has nothing to do with whether matrix values are numerically large or small.

Rank is about the number of independent directions/information dimensions in a matrix.

This is especially important when later studying LoRA.

---

# 9. Mental Models

## 9.1 Quantization Mental Model

Imagine a ruler.

Suppose you have a ruler with:

```text
1,000 tiny measurement marks
```

but your application only needs:

```text
16 marks
```

You can represent the values much more compactly.

But measurements become less precise.

Therefore:

```text
More bits
→ more numerical choices
→ more precision
→ more memory

Fewer bits
→ fewer choices
→ approximation
→ less memory
```

---

# 9.2 Attention Mental Model

Imagine a classroom with \(N\) students.

## Full Attention

Every student talks to every other student.

Number of interactions:

\[
O(N^2)
\]

---

## Sliding Window

Each student talks only to nearby students.

Number of interactions:

\[
O(NW)
\]

---

## Sparse Attention

Each student talks to selected students.

Only chosen connections are made.

---

## GQA

Students still attend to the context, but groups of students share the same key/value information.

This reduces stored information.

---

## FlashAttention

The conversations still happen, but instead of writing every intermediate conversation to a slow notebook, the system processes blocks efficiently using fast memory.

---

# 9.3 Scaling Laws Mental Model

Imagine training students.

You have a fixed education budget.

You can spend it on:

```text
more intelligent student
```

or:

```text
more books/practice
```

A very intelligent student with almost no practice:

```text
under-trained
```

A smaller student with enormous practice:

```text
more data per capacity
```

Scaling laws ask:

> What balance uses the budget most effectively?

---

# 9.4 Quantization + Attention + Scaling

Think of an LLM as a factory.

### Scaling Laws

Decide:

> **How large should the factory be and how much raw material should it process during training?**

### Attention

Decide:

> **How should information flow between pieces of context?**

### Quantization

Decide:

> **How compactly should the factory's numerical machinery be stored during deployment?**

---

# 10. Final Revision Checklist

Before considering these three topics mastered, make sure you can explain all of the following without memorizing blindly.

## Quantization

- [ ] Why quantization is needed
- [ ] What a bit represents
- [ ] FP32
- [ ] FP16
- [ ] BF16
- [ ] FP8
- [ ] INT8
- [ ] INT4
- [ ] Sign/exponent/mantissa
- [ ] Exponent vs mantissa
- [ ] Quantization
- [ ] Dequantization
- [ ] Scale
- [ ] Zero point
- [ ] Symmetric quantization
- [ ] Asymmetric quantization
- [ ] Per-tensor quantization
- [ ] Per-channel quantization
- [ ] Quantization error
- [ ] MSE
- [ ] Weight quantization
- [ ] Activation quantization
- [ ] KV-cache quantization
- [ ] PTQ
- [ ] QAT
- [ ] Fake quantization
- [ ] STE
- [ ] GPTQ
- [ ] Hessian intuition
- [ ] AWQ
- [ ] Activation-aware protection
- [ ] GGUF
- [ ] Q4_K_M
- [ ] Quantization vs lossless compression
- [ ] Memory calculations
- [ ] Perplexity and benchmark evaluation

---

## Attention

- [ ] Q, K, V
- [ ] Scaled dot-product attention
- [ ] Why divide by \(\sqrt{d_k}\)
- [ ] Softmax
- [ ] Causal masking
- [ ] \(O(N^2)\)
- [ ] Why doubling context quadruples dense interactions
- [ ] Sliding Window Attention
- [ ] \(O(NW)\)
- [ ] Long-range information propagation
- [ ] Sparse attention
- [ ] Block attention
- [ ] Local attention
- [ ] Global attention
- [ ] Random attention
- [ ] Strided attention
- [ ] \(O(N\sqrt N)\) structured sparse intuition
- [ ] Differential Attention
- [ ] \(A_1-\lambda A_2\)
- [ ] Attention sinks
- [ ] MHA
- [ ] MQA
- [ ] GQA
- [ ] KV cache
- [ ] KV-cache memory formula
- [ ] Why GQA reduces KV memory
- [ ] FlashAttention
- [ ] FlashAttention vs sparse attention
- [ ] FlashAttention vs SWA
- [ ] GQA vs SWA

---

## Scaling Laws

- [ ] Parameters \(N\)
- [ ] Tokens \(D\)
- [ ] Training compute \(C\)
- [ ] \(C\approx6ND\)
- [ ] Bigger model vs more data
- [ ] GPT-3 example
- [ ] Under-training
- [ ] Chinchilla
- [ ] ~20 tokens/parameter
- [ ] Hoffmann loss law
- [ ] \(A/N^\alpha\)
- [ ] \(B/D^\beta\)
- [ ] \(E\)
- [ ] Diminishing returns
- [ ] Fixed-compute optimization
- [ ] Why \(D=C/(6N)\)
- [ ] Differentiation idea
- [ ] Compute-optimal scaling
- [ ] \(N,D\propto\sqrt C\) under the approximate linear-ratio setting
- [ ] Training-optimal vs inference-aware scaling
- [ ] Over-training terminology
- [ ] High token/parameter ratios
- [ ] Emergence and evaluation artifacts
- [ ] Data quality
- [ ] MoE active vs total parameters
- [ ] Post-training
- [ ] Multimodal scaling
- [ ] Synthetic data
- [ ] Optimizer/effective compute

---

# Final One-Page Mental Summary

If you remember only one page, remember this:

## 1. Quantization

LLMs contain huge numbers of parameters.

Instead of storing:

\[
\text{FP16}
\]

we can use:

\[
\text{INT8/INT4}
\]

to reduce memory.

Core idea:

\[
q=\operatorname{round}(x/s)
\]

\[
\hat{x}=qs
\]

Main trade-off:

```text
Fewer bits
    ↓
Less memory
    ↓
Potentially faster/cheaper inference
    ↓
More approximation error
```

---

## 2. Attention Variants

Normal attention:

\[
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

has:

\[
O(N^2)
\]

dense interactions.

Different techniques attack different bottlenecks:

```text
SWA
→ fewer local interactions

Sparse Attention
→ selected interactions only

Differential Attention
→ subtract attention patterns

MQA/GQA
→ smaller KV cache

FlashAttention
→ more efficient implementation/memory traffic
```

---

## 3. Scaling Laws

Training depends strongly on:

\[
N=\text{parameters}
\]

\[
D=\text{training tokens}
\]

and approximately:

\[
C\approx6ND
\]

Chinchilla-style compute-optimal training gives approximately:

\[
D\approx20N
\]

and therefore, in that simplified regime:

\[
N_{\text{opt}}\propto\sqrt C
\]

\[
D_{\text{opt}}\propto\sqrt C
\]

Main lesson:

> **Do not think only about model size. Think about model size + training data + compute together.**

---

# The Three Topics in One Sentence

> **Scaling Laws tell you how much model and data to train, Attention Variants tell you how to process context efficiently, and Quantization tells you how to represent the trained model using fewer bits for efficient deployment.**

