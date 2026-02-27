"""
Tensor shape type aliases used across the TimesNet module.

This file defines shorthand tensor annotations based on torchtyping.TensorType
to improve readability and enforce consistent shape contracts in the codebase.

The aliases are purely for static/type-checking and documentation purposes.
They do not introduce runtime overhead or modify tensor behavior.

Defined aliases
---------------
BTC
    Tensor with shape [B, T, C]
    B = batch size
    T = time / sequence length
    C = feature / channel dimension

BTD
    Tensor with shape [B, T, D]
    D = model / embedding dimension

BTK
    Tensor with shape [B, T, K]
    K = number of temporal/calendar features

BCRP
    Tensor with shape [B, C, Rows, Period]
    Used for folded 2D representations in TimesNet blocks

BCHW
    Tensor with shape [B, C, H, W]
    Standard Conv2D feature map layout
"""

from torchtyping import TensorType

# Common shorthand tensor shapes
BTC = TensorType["b", "t", "c"]  # batch, time, channels
BTD = TensorType["b", "t", "d"]  # batch, time, d_model
BTK = TensorType["b", "t", "k"]  # batch, time, discrete calendar features
BCRP = TensorType["b", "c", "rows", "p"]  # folded 2D
BCHW = TensorType["b", "c", "h", "w"]  # generic conv2d map
