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



