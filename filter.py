import torch

from .core.filters import KuwaharaFilter


class Filter:
    CATEGORY = "💫PixelUtils/Filters"


class KuwaharaNode(Filter):
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
        return {
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
                            "The radius of the sliding window. \n"
                            "1 (Weak): Removes pixel noise, keeps tiny details. \n"
                            "2-3 (Balanced): Best for Pixel Art. Creates a clean 'painterly' look. \n"
                            "4+ (Strong): Very abstract, blocky, and loses shape definitions."
                        ),
                    },
                ),
            },
            "optional": {
                # image: Optional input for real-time previewing.
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional input image for previewing the filter effect."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "PX_FILTER")
    RETURN_NAMES = ("image", "px_filter")
    FUNCTION = "get_filter"

    def get_filter(self, radius, image=None):
        # Instantiate the filter wrapper (logic carrier)
        filter_obj = KuwaharaFilter(radius)

        # Preview logic: Apply the filter immediately if an image is provided
        preview_result = None
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
