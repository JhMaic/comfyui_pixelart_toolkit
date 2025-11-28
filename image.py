import numpy as np
import torch

from .core.pyxelate import Pyx


class Image:
    CATEGORY = "💫PixelUtils/Image"


class PyxelateTransformNode(Image):
    # -------------------------------------------------------------------
    # Node Description
    # -------------------------------------------------------------------
    DESCRIPTION = """
    Advanced wrapper for Pyxelate with resolution priority logic.
    Also see: https://github.com/sedthh/pyxelate/tree/master

    Logic:
    - If 'factor' > 0: The image is downsampled by 1/factor (Height & Width inputs are IGNORED).
    - If 'factor' == 0: The image is resized to 'height' and 'width'.
    """

    title = "Pyxelate Transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "height": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Target height. Used ONLY if factor is 0.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Target width. Used ONLY if factor is 0.",
                    },
                ),
                "downscale_factor": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Downsample factor (e.g. 4 = 1/4 size). Set to 0 to use Height/Width instead.",
                    },
                ),
                # Description: Resizes the pixels of the transformed image by upscale.
                # Default is 1.
                "upscale_factor": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Resizes the pixels of the transformed image by upscale. Default is 1 (do nothing). It will overwrite Height/Width if it is > 1.",
                    },
                ),
                # Description: The number of colors in the transformed image.
                # If int > 2, searches automatically. Default is 32.
                "palette": (
                    "INT",
                    {
                        "default": 32,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                        "tooltip": "The number of colors in the transformed image. Pyxelate will search for this many colors automatically if int > 2. Default is 32.",
                    },
                ),
                # Description: The type of dithering to use on the transformed image.
                # Options: none, naive, bayer, floyd, atkinson.
                "dither": (
                    ["none", "naive", "bayer", "floyd", "atkinson"],
                    {
                        "default": "none",
                        "tooltip": "The type of dithering to use. 'none' (default), 'naive' (for alpha), 'bayer' (fastest), 'floyd' (slowest), 'atkinson' (slowest).",
                    },
                ),
                # Description: For images with transparency, the transformed image's pixel will be either visible/invisible above/below this threshold.
                # Default is 0.6.
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "For images with transparency, the transformed image's pixel will be either visible/invisible above/below this threshold. Default is 0.6.",
                    },
                ),
                # Description: The size of the sobel operator (N*N area to calculate the gradients for downsampling).
                # Must be an int larger than 1. Default is 3.
                "sobel": (
                    "INT",
                    {
                        "default": 3,
                        "min": 2,
                        "max": 15,
                        "step": 1,
                        "tooltip": "The size of the sobel operator (N*N area to calculate the gradients for downsampling), must be an int larger than 1. Default is 3, try 2 for a much faster but less accurate output.",
                    },
                ),
                # Description: How many times should the Pyxelate algorithm be applied to downsample the image.
                # Default is 1. Should never be more than 3.
                "depth": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "How many times should the Pyxelate algorithm be applied to downsample the image. More iteratrions will result in blockier aesthatics. Must be a positive int, although it is really time consuming and should never be more than 3. Raise it only for really small images. Default is 1",
                    },
                ),
            },
            "optional": {
                "px_filter": ("PX_FILTER",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    def process(
        self,
        image,
        downscale_factor,
        height,
        width,
        upscale_factor,
        palette,
        dither,
        alpha,
        sobel,
        depth,
        px_filter=None,
    ):

        results = []

        # Logic Determination
        # We prepare arguments once, as they apply to the whole batch based on settings
        if downscale_factor > 0:
            # Mode A: Use Factor (Ignore H/W)
            # We explicitly set h/w to None so Pyxelate logic purely uses factor
            arg_factor = downscale_factor
            arg_height = None
            arg_width = None
        else:
            # Mode B: Use Height/Width (Ignore Factor)
            # We pass factor as None (or Pyxelate won't calculate scale based on dimensions)
            arg_factor = None

            # Handle 0 dimensions: If user sets everything to 0, fallback to factor=1 (original size)
            if height == 0 and width == 0:
                arg_factor = 1
                arg_height = None
                arg_width = None
            else:
                arg_height = height if height > 0 else None
                arg_width = width if width > 0 else None

            if upscale_factor > 0:
                arg_factor = 1
                arg_height = None
                arg_width = None

        for tensor_img in image:
            # Tensor (0.0-1.0) -> Numpy uint8 (0-255)
            img_np = (tensor_img.numpy() * 255).astype(np.uint8)

            # Instantiate Pyx
            pyx_transformer = Pyx(
                height=arg_height,
                width=arg_width,
                factor=arg_factor,
                upscale=upscale_factor,
                palette=palette,
                dither=dither,
                alpha=alpha,
                sobel=sobel,
                depth=depth,
                filter_obj=px_filter,
            )

            # Transform
            processed_result = pyx_transformer.fit_transform(img_np)

            if isinstance(processed_result, torch.Tensor):
                processed_tensor = processed_result.detach().cpu().squeeze(0)
            else:
                processed_tensor = torch.from_numpy(
                    processed_result.astype(np.float32) / 255.0
                )

            results.append(processed_tensor)

        return (torch.stack(results),)
