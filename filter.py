from abc import ABC, abstractmethod
from typing import Optional, override

import torch

from .core.filters import KuwaharaFilter, FilterWrapper, MorphFilter


class FilterNode(ABC):
    CATEGORY = "💫PixelToolkit/Filters"

    RETURN_TYPES = ("IMAGE", "PX_FILTER")
    RETURN_NAMES = ("image", "px_filter")
    FUNCTION = "exec"

    @classmethod
    def build_inputs(cls, specific_inputs: dict):
        if "required" not in specific_inputs:
            specific_inputs["required"] = {}

        if "optional" not in specific_inputs:
            specific_inputs["optional"] = {}

        specific_inputs["optional"].update(
            {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional. Input image for preview chaining. Output Image is available only if this is set."
                    },
                ),
                "px_filter": (
                    "PX_FILTER",
                    {"tooltip": "Optional. Previous filter to chain with."},
                ),
            }
        )

        return specific_inputs

    @abstractmethod
    def init_filter(self, **kwargs) -> FilterWrapper:
        pass

    def exec(
        self,
        image: Optional[torch.Tensor] = None,
        px_filter: Optional[FilterWrapper] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, FilterWrapper]:
        # Instantiate the filter wrapper (logic carrier)
        filter_obj = self.init_filter(**kwargs)

        # If it has previous filter, set it.
        if px_filter:
            filter_obj.set_previous_filter(px_filter)

        preview_result = None
        # Preview logic: Apply the filter immediately if an image is provided
        if image is not None:
            # Ensure processing happens on the correct device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Convert ComfyUI format [B, H, W, C] to PyTorch format [B, C, H, W]
            img_t = image.permute(0, 3, 1, 2).to(device)

            # Apply the filter logic
            processed = filter_obj(img_t)

            # Convert back to ComfyUI format [B, H, W, C] and move to CPU
            preview_result = processed.permute(0, 2, 3, 1).cpu()

        # Return the preview (or None) and the filter object itself
        return (preview_result, filter_obj)


class KuwaharaNode(FilterNode):
    """
    Kuwahara Filter Node for Pyxelate.

    This node creates a Kuwahara filter configuration. The Kuwahara filter is a non-linear
    smoothing filter that preserves edges while flattening the internal areas of shapes.

    For Pixel Art:
    It is extremely effective at removing "dirty" noise and converting smooth gradients
    (like a blue sky) into distinct, solid color blocks. This makes the subsequent
    palette quantization step much cleaner and more accurate.
    """

    TITLE = "Kuwahara Filter"

    # Description shown in ComfyUI manager or node info
    DESCRIPTION = """
    Creates an edge-preserving smoothing filter (Kuwahara) for the Pyxelate pipeline.

    This filter transforms the image into an "oil painting" style by smoothing colors 
    while keeping outlines sharp. It is highly recommended for pixel art to prevent 
    dithering noise and to help cluster colors effectively.

    Inputs:
    - radius: Controls the window size of the filter.
    - image (Optional): Connect an image here to preview the filter effect immediately.
    """

    @classmethod
    def INPUT_TYPES(s):
        return s.build_inputs(
            {
                "required": {
                    # radius: The most important parameter.
                    # A value of 2 corresponds to a 5x5 scanning window.
                    "radius": (
                        "INT",
                        {
                            "default": 2,
                            "min": 1,
                            "max": 10,
                            "step": 1,
                            "tooltip": (
                                "The window radius. E.g., 2 specifies a 5x5 window ((2*2)+1). \n"
                                "1 (Weak): Removes pixel noise, keeps tiny details. \n"
                                "2-3 (Balanced): Best for Pixel Art. Creates a clean 'painterly' look. \n"
                                "4+ (Strong): Very abstract, blocky, and loses shape definitions."
                            ),
                        },
                    ),
                },
            }
        )

    RETURN_TYPES = ("IMAGE", "PX_FILTER")
    RETURN_NAMES = ("image", "px_filter")

    @override
    def init_filter(self, radius: int) -> FilterWrapper:
        return KuwaharaFilter(radius)


class MorphNode(FilterNode):
    """
    Morphological Filter Node.
    Uses mathematical morphology (Erosion/Dilation) to clean up noise.
    """

    TITLE = "Morph Filter"

    DESCRIPTION = """
    Applies Morphological operations to remove small noise patches.

    - Open: Removes bright spots (salt noise) on dark background.
    - Close: Removes dark spots (pepper noise) on bright background.
    - Both: Removes both types (Strong cleaning).

    """

    @classmethod
    def INPUT_TYPES(s):
        return s.build_inputs(
            {
                "required": {
                    "operation": (
                        ["open", "close", "both", "dilate", "erode"],
                        {"default": "both"},
                    ),
                    "kernel_size": (
                        "INT",
                        {
                            "default": 3,
                            "min": 3,
                            "max": 15,
                            "step": 2,
                            "tooltip": "Kernel size (odd number). 3 removes 1px dots, 5 removes larger blobs.",
                        },
                    ),
                },
            }
        )

    @override
    def init_filter(self, operation: str, kernel_size: int) -> FilterWrapper:
        # Pass parameters to the logic class
        return MorphFilter(operation, kernel_size)
