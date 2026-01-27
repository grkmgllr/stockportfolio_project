from torchtyping import TensorType

# Common shorthand tensor shapes
BTC = TensorType["b", "t", "c"]  # batch, time, channels
BTD = TensorType["b", "t", "d"]  # batch, time, d_model
BTK = TensorType["b", "t", "k"]  # batch, time, discrete calendar features
BCRP = TensorType["b", "c", "rows", "p"]  # folded 2D
BCHW = TensorType["b", "c", "h", "w"]  # generic conv2d map
