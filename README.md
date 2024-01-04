# A numpy based ML framework.

Implement basic functionalities of ML frameworks using Numpy. The goal of this
project is to help build understanding of basic building blocks of modeling
and training and the math behind it. Nothing practical beyond that.

## Features

### `layers`

Supports forward and backward path. Tested using Jax/Flax.

- layers
    - `Dense`: Fully connected layer.
    - `Conv2D`: 2D Convolutional layer.
    - `MultiHeadAttention`: Attention mechanism. Found in transformer
    encoder/decoder blocks.
    - `TransformerEncoder`: Transformer decoder block. Found in encoder-only
    architecture (BERT), encoder-decoder architecture (BART, T5).
    - `TransformerDecoder`: Transformer decoder block. Found in decoder-only
    arhictecture (GPT, PaLM, LLaMA), encoder-decoder architecture (BART, T5).

- `activaitons`
    - `ReLU`: [ReLU activation](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
        Basic and popular non-linear activation.
    - `Softmax`: [Softmax activation](https://en.wikipedia.org/wiki/Softmax_function).
        Normalize output as a probability distribution.

- `normalizations`
    - `Dropout`: [Dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf).
        Resolve overfitting through preventing units co-adopting.
    - `LayerNormalization`: [Layer normalization](https://arxiv.org/abs/1607.06450).
        Normalize each individual sample in a batch. Common in autoregressive
        NLP tasks.

### `loss`

- `MSELoss`: Mean square error loss for regression tasks.

### `optimizer`

- `SGDOptimizer`: Stochastic gradient descent.
- `AdamOptimizer`: [Adam optimizer](https://arxiv.org/abs/1412.6980).
    Dynamically adjusting learning rate on individual weights based on
    momentumn and velocity.

### `train`

- `Trainer`: Naive local trainer.