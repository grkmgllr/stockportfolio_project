import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """
    Inception-style convolution block.

    Applies multiple kernel sizes in parallel to capture
    patterns at different temporal scales.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()

        kernels = []
        for k in range(num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * k + 1,
                    padding=k
                )
            )

        self.convs = nn.ModuleList(kernels)

    def forward(self, x):
        # Sum outputs of all kernels
        return sum(conv(x) for conv in self.convs) / len(self.convs)