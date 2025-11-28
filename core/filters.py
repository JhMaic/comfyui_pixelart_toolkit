from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn.functional as F


class FilterWrapper(ABC):
    def __init__(self):
        self._previous_filter = None

    # Ensure it accepts [B, C, H, W] format
    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self._previous_filter is None:
            result = self.process(image_tensor)
        else:
            prev_result = self._previous_filter(image_tensor)
            result = self.process(prev_result)

        return result

    def set_previous_filter(self, previous_filter: "FilterWrapper") -> None:
        self._previous_filter = previous_filter

    @abstractmethod
    def process(self, image_tensor) -> torch.Tensor:
        pass


class KuwaharaFilter(FilterWrapper):
    def __init__(self, radius: int = 2):
        super().__init__()
        self.radius = radius

    @override
    def process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kuwahara filter GPU implementation (best smoothing algorithm for pixel art)
        radius: The window radius. E.g., 2 specifies a 5x5 window ((2*2)+1).
        """
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # Kernel size
        kernel_size = 2 * self.radius + 1

        # 1. Precompute Mean and Mean of Squares
        # Use average pooling as a Box Filter
        # We need to calculate for four quadrants separately, but using a large convolution with padding is more efficient
        # Here, for clarity and versatility, we use Unfold or simulate with average pooling

        # To avoid loops, we use pad and avg_pool to calculate the four regions
        # Region definition: TL (Top-Left), TR (Top-Right), BL (Bottom-Left), BR (Bottom-Right)
        # The size of each region is (radius + 1)

        # First, pad the image to handle edges
        x_pad = F.pad(
            x, (self.radius, self.radius, self.radius, self.radius), mode="reflect"
        )

        # Define a simple function for mean calculation (Unused but kept for context)
        def get_mean_std(img_pad, k_size):
            # Mean
            mean = F.avg_pool2d(
                img_pad, kernel_size=(k_size, k_size), stride=1, padding=0
            )
            # Mean of Squares
            sq_mean = F.avg_pool2d(
                img_pad**2, kernel_size=(k_size, k_size), stride=1, padding=0
            )
            # Variance = E[X^2] - (E[X])^2
            variance = sq_mean - mean**2
            return mean, variance

        # Since Kuwahara requires four corner windows centered around the current pixel
        # This can be achieved by performing mean filtering on the entire padded image once, and then slicing (offsetting) the result.

        # This is an optimized implementation: calculating the local mean and variance of the large padded image once, and then slicing out the four corners
        # This method differs slightly from standard Kuwahara (due to overlapping regions), but the effect is consistent and fast

        # Window size (radius + 1)
        w_size = self.radius + 1

        # Calculate the sliding window mean and variance for the entire padded image
        # The avg_pool window here is the size of the sub-region
        mean_all = F.avg_pool2d(x_pad, kernel_size=w_size, stride=1)
        sq_mean_all = F.avg_pool2d(x_pad**2, kernel_size=w_size, stride=1)
        var_all = sq_mean_all - mean_all**2

        # The dimension of mean_all is now approximately [B, C, H+radius, W+radius]
        # We need to slice out the four corresponding regions
        # The (0,0) point of the original image is at (radius, radius) in x_pad
        # The center of the corresponding sub-window mean needs to be offset

        # Top-Left: Covers [y-r, x-r] to [y, x]
        # In mean_all, corresponds to indices 0:h, 0:w
        m0 = mean_all[:, :, 0:h, 0:w]
        v0 = var_all[:, :, 0:h, 0:w]

        # Top-Right: Covers [y-r, x] to [y, x+r]
        # In mean_all, corresponds to indices 0:h, r:w+r
        m1 = mean_all[:, :, 0:h, self.radius : w + self.radius]
        v1 = var_all[:, :, 0:h, self.radius : w + self.radius]

        # Bottom-Left: Covers [y, x-r] to [y+r, x]
        # In mean_all, corresponds to indices r:h+r, 0:w
        m2 = mean_all[:, :, self.radius : h + self.radius, 0:w]
        v2 = var_all[:, :, self.radius : h + self.radius, 0:w]

        # Bottom-Right: Covers [y, x] to [y+r, x+r]
        # In mean_all, corresponds to indices r:h+r, r:w+r
        m3 = mean_all[
            :, :, self.radius : h + self.radius, self.radius : w + self.radius
        ]
        v3 = var_all[:, :, self.radius : h + self.radius, self.radius : w + self.radius]

        # 2. Find the index of the minimum variance
        # Sum the variance (across channels, total RGB variance), because we want RGB to select the same region as a whole
        # [B, 1, H, W]
        v0_sum = torch.sum(v0, dim=1, keepdim=True)
        v1_sum = torch.sum(v1, dim=1, keepdim=True)
        v2_sum = torch.sum(v2, dim=1, keepdim=True)
        v3_sum = torch.sum(v3, dim=1, keepdim=True)

        # Stack variances [B, 4, H, W]
        stack_var = torch.cat([v0_sum, v1_sum, v2_sum, v3_sum], dim=1)

        # Find the index of the minimum variance [B, 1, H, W]
        best_idx = torch.argmin(stack_var, dim=1, keepdim=True)

        # 3. Get the corresponding mean based on the index
        # Expand mask to RGB channels [B, 3, H, W]
        mask = best_idx.repeat(1, c, 1, 1)

        # Stack means [B, 4, C, H, W] -> [B, C, H, W, 4] for easier gather operation
        stack_mean = torch.stack([m0, m1, m2, m3], dim=4)

        # Use gather to retrieve values
        # gather requires index dimension matching; we are gathering along dim=4
        # Reshape mask to [B, C, H, W, 1]
        mask_gather = mask.unsqueeze(-1)

        output = torch.gather(stack_mean, 4, mask_gather).squeeze(-1)

        return output
