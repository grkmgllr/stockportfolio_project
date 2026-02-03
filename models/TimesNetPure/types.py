"""
Tensor shape aliases used across the TimesNetPure implementation.

This module defines common shorthand tensor types using `torchtyping.TensorType`
to document and enforce expected tensor shapes throughout the codebase.

The aliases are purely for type annotation and readability; they do not affect
runtime behavior.

Conventions
-----------
All tensor shapes follow the [B, T, C]-style notation:
    - B: batch size
    - T: time dimension
    - C: channel / feature dimension
    - D: model (embedding) dimension
    - K: number of discrete time features
    - H, W: spatial dimensions for 2D feature maps

These aliases are used in function signatures and docstrings to make shape
contracts explicit and self-documenting.
"""

from torchtyping import TensorType

# Common shorthand tensor shapes
BTC = TensorType["b", "t", "c"]  # batch, time, channels
BTD = TensorType["b", "t", "d"]  # batch, time, d_model
BTK = TensorType["b", "t", "k"]  # batch, time, discrete calendar features
BCRP = TensorType["b", "c", "rows", "p"]  # folded 2D
BCHW = TensorType["b", "c", "h", "w"]  # generic conv2d map
