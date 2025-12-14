import torch
import torch.nn as nn

class TimeImageDecomposition(nn.Module):
    """
    TID: Decomposes 2D Time Images into Season and Trend Images.
    Uses 2D Average Pooling to capture 'Image Trend'.
    """
    def __init__(self, kernel_size=(3, 3)):
        super().__init__()
        # We use odd kernel sizes to maintain symmetry
        # Padding handles borders so output size == input size
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        
        self.moving_avg_2d = nn.AvgPool2d(
            kernel_size=kernel_size, 
            stride=1, 
            padding=(pad_h, pad_w), 
            count_include_pad=False
        )

    def forward(self, x_image):
        """
        x_image: [Batch, Channel, Height, Width]
        """
        # 1. Extract Trend Image (The smooth background)
        trend_image = self.moving_avg_2d(x_image)
        
        # 2. Extract Season Image (The details/texture)
        season_image = x_image - trend_image
        
        return season_image, trend_image