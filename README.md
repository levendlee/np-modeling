# A numpy based ML framework.

Implement basic functionalities of ML frameworks using Numpy. The goal of this
project is to help build understanding of basic building blocks of modeling
and training and the math behind it. Nothing practical beyond that.

## Features

### Layers

Supports forward and backward path. Tested using Jax/Flax.

- `mlp.Dense`
- `conv.Conv`
- `attention.MultiHeadAttention`
- `transformer.TransformerEncoder`
- `transformer.TransformerDecoder`

- `mlp.ReLU`
- `attention.Softmax`

- `transformer.LayerNormalization`

### Loss

- `loss.MSELoss`

### Optimizer

- `optimizer.AdamOptimizer`

### Training

- `train.Trainer`

