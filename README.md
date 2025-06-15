# Higher-Order Transformers

This project explores higher-order attention mechanisms in transformer models, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Motivation

Standard transformer models, through their self-attention mechanism, are fundamentally designed to compute relationships between *pairs* of tokens. While powerful, this pairwise attention has a limitation: to model more complex, higher-order relationships between *tuples* of tokens (e.g., how three or more tokens relate to each other simultaneously), the model must rely on depth. It needs to stack many layers, with each subsequent layer combining the pairwise relationships from the one before it.

This is a computationally inefficient, brute-force approach to capturing complex dependencies in data. The depth of the network becomes the primary, and very costly, axis for computing these richer representations.

## Project Goal

This project explores an alternative: what if we could compute these higher-order relationships directly within a single attention layer? The goal is to implement and experiment with a *higher-order attention mechanism*.

By explicitly modeling n-ary relationships between tokens, we aim to unlock a new computational axis for transformers. This could potentially lead to models that are more computationally efficient and powerful, capable of capturing complex data structures without requiring excessive depth.

## Implementation

We use the minimal and efficient nanoGPT implementation as our foundation, modifying key components to incorporate higher-order attention. The code is designed to be as lightweight and readable as possible while enabling meaningful experiments.

### Higher-Order Attention Formulation

Standard self-attention computes pairwise interactions. For a query token \(q_i\) and a key token \(k_j\), the attention score is based on their dot product. Our higher-order attention mechanism extends this to capture interactions between tuples of tokens.

Given an input sequence, for each token, we compute one query vector \(q_i\) and \(n-1\) sets of key and value vectors, \(\{k^m_i\}_{m=1}^{n-1}\) and \(\{v^m_i\}_{m=1}^{n-1}\), where \(n\) is the order of attention. For a single head with hidden dimension \(d\), the formulation is as follows:

1.  **Attention Scores**: The score for a query token \(i\) and a tuple of \(n-1\) key tokens \((j_1, \dots, j_{n-1})\) is computed via a sum-product over the head dimension \(d\):
    \[
    \text{score}(i, j_1, \dots, j_{n-1}) = \frac{1}{\sqrt{d}} \sum_{l=1}^{d} q_{il} \cdot k^1_{j_1 l} \cdot \dots \cdot k^{n-1}_{j_{n-1} l}
    \]
    This can also be expressed using the Hadamard (element-wise) product \(\odot\):
    \[
    \text{score}(i, j_1, \dots, j_{n-1}) = \frac{1}{\sqrt{d}} \mathbf{1}^T (q_i \odot k^1_{j_1} \odot \dots \odot k^{n-1}_{j_{n-1}})
    \]

2.  **Attention Weights**: The weights are obtained by applying a softmax over all possible key-token tuples for each query token. Causal masking is applied to prevent attending to future tokens.
    \[
    A_{i, j_1, \dots, j_{n-1}} = \text{softmax}_{j_1, \dots, j_{n-1}} \left( \text{score}(i, j_1, \dots, j_{n-1}) \right)
    \]

3.  **Aggregated Values**: The value vector for a key-tuple is an element-wise product of their individual value vectors:
    \[
    V_{j_1, \dots, j_{n-1}} = v^1_{j_1} \odot v^2_{j_2} \odot \dots \odot v^{n-1}_{j_{n-1}}
    \]

4.  **Output**: The final output for token \(i\) is a weighted sum over the aggregated values:
    \[
    y_i = \sum_{j_1, \dots, j_{n-1}} A_{i, j_1, \dots, j_{n-1}} \cdot V_{j_1, \dots, j_{n-1}}
    \]

This formulation allows a single attention head to directly model n-ary relationships between tokens in a sequence.

## Dataset Preparation

For experimenting with mixed curvature transformers, we use the same dataset preparation approach as the original nanoGPT:

### Shakespeare Dataset (Small Scale Testing)

For quick experimentation, the Shakespeare dataset provides a lightweight option:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin` files with character-level tokenization.

### OpenWebText Dataset (Full Scale Training)

For more extensive training, prepare the OpenWebText dataset:

```sh
python data/fineweb/prepare.py
```

This downloads and tokenizes the fineweb dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

Both datasets are prepared to be used with the training scripts. For mixed curvature experiments, we can use these datasets to compare performance against baseline Euclidean transformer architectures.

## Getting Started

[Installation and usage instructions will be added as the project develops]

## Acknowledgements

- Original code based on on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy



