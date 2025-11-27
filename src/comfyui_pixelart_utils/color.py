import torch


class Color:
    CATEGORY = "💫PixelUtils/Color"


class PickPixelColorAtPosition(Color):
    title = "Pick Color At Position"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "x": ("INT", {"min": 0, "default": 0, "display": "number"}),
                "y": ("INT", {"min": 0, "default": 0, "display": "number"}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = (
        "rgb_int",
        "rgb_hex",
        "rbga_hex",
    )
    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, x: int, y: int):
        B, H, W, C = image.shape

        if x < 0 or x >= W or y < 0 or y >= H:
            raise ValueError(
                f"Coordinate (x={x}, y={y}) is out of bounds for image size (W={W}, H={H})"
            )

        pixel_values = image[:, y, x, :]

        pixel_values = pixel_values.detach().cpu()

        if torch.is_floating_point(pixel_values):
            if pixel_values.max() <= 1.0:
                pixel_values = pixel_values * 255.0

            pixel_values = pixel_values.round().clamp(0, 255).long()
        else:
            pixel_values = pixel_values.clamp(0, 255).long()

        rgb_list = pixel_values[0].tolist()
        rgb_hex = "{:02X}{:02X}{:02X}".format(rgb_list[0], rgb_list[1], rgb_list[2])
        rgba_hex = "{:02X}{:02X}{:02X}{:02X}".format(
            rgb_list[0],
            rgb_list[1],
            rgb_list[2],
            rgb_list[3] if len(rgb_list) != 3 else 0xFF,
        )
        rgb_int = int(rgb_hex, 16)

        return (rgb_int, f"#{rgb_hex}", f"#{rgba_hex}")


import numpy as np
from PIL import Image


class PixelArtReduceColors(Color):
    DESCRIPTION = """
       Reduces the color palette of an image (RGB or RGBA) to a fixed number of colors.
       Specifically optimized for Pixel Art.

       Key Features:
       - RGBA Support: Alpha channel is preserved.
       - Alpha Threshold: Converts semi-transparent edges to hard pixel edges (1-bit alpha).

       Parameters:
       - max_colors: Target number of colors (e.g., 16, 32).
       - algo_mode: Quantization algorithm (median_cut is standard).
       - dither: "disabled" is recommended for clean pixel art; "floyd_steinberg" adds noise.
       - alpha_threshold: 
         - 0.0: Keep alpha exactly as is (soft edges).
         - 0.5: Standard cutoff (pixels < 0.5 opacity become fully invisible).
         - 1.0: (Not recommended) might remove everything.
       """
    title = "Reduce Colors"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
                "algo_mode": (
                    ["fast_octree", "median_cut", "max_coverage"],
                    {"default": "max_coverage"},
                ),
                "dither": (["disabled", "floyd_steinberg"],),
                "alpha_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 = Keep original alpha. > 0.0 = Hard cutout threshold.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce_colors"

    def reduce_colors(self, image, max_colors, algo_mode, dither, alpha_threshold):
        # image shape: [Batch, H, W, C]
        result_images = []

        # Map algo modes
        method_map = {
            "median_cut": Image.Quantize.MEDIANCUT,
            "max_coverage": Image.Quantize.MAXCOVERAGE,
            "fast_octree": Image.Quantize.FASTOCTREE,
        }
        pil_method = method_map.get(algo_mode, Image.Quantize.MEDIANCUT)

        # Map dither
        dither_val = (
            Image.Dither.NONE if dither == "disabled" else Image.Dither.FLOYDSTEINBERG
        )

        for img_tensor in image:
            # 1. Detect Channels
            height, width, channels = img_tensor.shape

            is_rgba = channels == 4
            img_alpha = None

            # 2. Separate Alpha if exists
            if is_rgba:
                # Split RGB and Alpha
                # RGB: [H, W, 3], Alpha: [H, W, 1]
                img_rgb_tensor = img_tensor[:, :, :3]
                img_alpha_tensor = img_tensor[:, :, 3:4]  # Keep dim for later concat
            else:
                img_rgb_tensor = img_tensor

            # 3. Convert RGB part to Numpy -> PIL for quantization
            # Tensor (0-1 float) -> Numpy (0-255 uint8)
            i = 255.0 * img_rgb_tensor.cpu().numpy()
            img_np = np.clip(i, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # 4. Perform Quantization (on RGB only)
            quantized_img = pil_img.quantize(
                colors=max_colors,
                method=pil_method,
                kmeans=20,  # Optimization for better palette
                dither=dither_val,
            )

            # 5. Convert back to RGB Tensor
            rgb_img = quantized_img.convert("RGB")
            out_np = np.array(rgb_img).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)  # [H, W, 3]

            # 6. Recombine Alpha
            if is_rgba:
                if alpha_threshold > 0.0:
                    # Apply Hard Edge Thresholding (Binary Alpha)
                    # Anything below threshold becomes 0, anything above becomes 1
                    mask = (img_alpha_tensor >= alpha_threshold).float()
                    img_alpha_tensor = mask

                # Concatenate [H, W, 3] + [H, W, 1] -> [H, W, 4]
                final_img = torch.cat((out_tensor, img_alpha_tensor), dim=2)
            else:
                final_img = out_tensor

            result_images.append(final_img)

        # Re-stack Batch
        return (torch.stack(result_images),)


class ColorConverter(Color):
    # -------------------------------------------------------------------
    # Node Description
    # -------------------------------------------------------------------
    DESCRIPTION = """
    A strict color converter that accepts ONE input format and converts it to all others.

    CRITICAL: You must provide EXACTLY ONE input. 
    If you connect/fill multiple inputs, the node will Error out.

    Inputs:
    - rgb_int: 24-bit Integer (e.g., 16711680 for Red).
    - rgb_hex: Hex String #RRGGBB.
    - rgba_hex: Hex String #RRGGBBAA.

    Outputs:
    - rgb_int: (24-bit Integer, Alpha stripped).
    - rgb_hex: (#RRGGBB).
    - rgba_hex: (#RRGGBBAA).
    """

    title = "Color Converter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                # We use specific "empty" values to detect if an input is used
                "rgb_int": ("INT", {"default": -1, "min": -1, "max": 16777215}),
                "rgb_hex": ("STRING", {"default": "", "multiline": False}),
                "rgba_hex": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "rgb_int",
        "rgb_hex",
        "rgba_hex",
    )
    FUNCTION = "convert"

    def convert(self, rgb_int, rgb_hex, rgba_hex):
        # 1. Input Validation Logic
        # We check which inputs have changed from their "empty" default states
        inputs_provided = []

        if rgb_int != -1:
            inputs_provided.append("rgb_int")

        if rgb_hex != "":
            inputs_provided.append("rgb_hex")

        if rgba_hex != "":
            inputs_provided.append("rgba_hex")

        # 2. Strict Mutual Exclusivity Check
        if len(inputs_provided) > 1:
            error_msg = f"ColorConverter Error: Multiple inputs detected ({', '.join(inputs_provided)}). Please connect ONLY one."
            raise ValueError(error_msg)

        if len(inputs_provided) == 0:
            raise ValueError(
                "ColorConverter Error: No input provided. Please connect one source."
            )

        # 3. Processing
        source = inputs_provided[0]
        r, g, b, a = 0, 0, 0, 255  # Default RGBA

        try:
            if source == "rgb_int":
                # Handle Integer
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                a = 255  # Int input is assumed opaque

            elif source == "rgb_hex":
                # Handle RGB String
                clean = rgb_hex.strip().lstrip("#").upper()
                if len(clean) == 6:
                    r = int(clean[0:2], 16)
                    g = int(clean[2:4], 16)
                    b = int(clean[4:6], 16)
                    a = 255
                elif len(clean) == 3:
                    r = int(clean[0] * 2, 16)
                    g = int(clean[1] * 2, 16)
                    b = int(clean[2] * 2, 16)
                    a = 255
                else:
                    raise ValueError(f"Invalid RGB Hex format: {rgb_hex}")

            elif source == "rgba_hex":
                # Handle RGBA String
                clean = rgba_hex.strip().lstrip("#").upper()
                if len(clean) == 8:
                    r = int(clean[0:2], 16)
                    g = int(clean[2:4], 16)
                    b = int(clean[4:6], 16)
                    a = int(clean[6:8], 16)
                elif len(clean) == 6:
                    # Fallback if user plugged 6-digit hex into rgba slot
                    r = int(clean[0:2], 16)
                    g = int(clean[2:4], 16)
                    b = int(clean[4:6], 16)
                    a = 255
                else:
                    raise ValueError(f"Invalid RGBA Hex format: {rgba_hex}")

        except ValueError as e:
            # Re-raise conversion errors to UI
            raise ValueError(f"Color Conversion Failed: {str(e)}")

        # 4. Generate Outputs
        out_rgb_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
        out_rgba_hex = "#{:02X}{:02X}{:02X}{:02X}".format(r, g, b, a)
        out_rgb_int = (r << 16) | (g << 8) | b

        return (out_rgb_int, out_rgb_hex, out_rgba_hex)


class PixelArtShadowSoftener(Color):
    # -------------------------------------------------------------------
    # Node Description
    # -------------------------------------------------------------------
    DESCRIPTION = """
    Fixes "dirty" dark pixels in pixel art by blending them with their neighbors.
    Instead of removing black pixels, it colors them based on surroundings.

    Great for fixing AI artifacts where shadows are pure black/grey instead of hue-shifted colors.

    Parameters:
    - contrast_threshold: How much darker a pixel must be to be considered a "bad shadow".
      (Lower = affects more pixels, Higher = only affects pitch black pixels).
    - soften_strength: How much neighbor color to inject.
      - 1.0 = Completely replace black with dark neighbor color.
      - 0.5 = Mix black with neighbor color.
    - darkening_factor: Controls how dark the new shadow should be relative to neighbors.
      - 0.5 = The new shadow is 50% brightness of the surrounding area.
    """

    title = "Shadow Softener"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "soften_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "darkening_factor": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "soften_shadows"

    def soften_shadows(
        self, image, contrast_threshold, soften_strength, darkening_factor
    ):
        result_images = []

        for img_tensor in image:
            # Convert to numpy [H, W, C]
            img = img_tensor.cpu().numpy()
            H, W, C = img.shape

            # Prepare output
            out_img = img.copy()

            # Helper to calculate luminance
            # Luminance = 0.299*R + 0.587*G + 0.114*B
            def get_luminance(arr):
                return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

            luminance_map = get_luminance(img)

            # Pad image for neighbor checks (Reflect padding handles edges better)
            img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")

            # Iterate pixels (Simple loop is fine for 64x64, vectorized is complex due to logic)
            # For larger images, this should be optimized with convolution, but for pixel art icons it's instant.
            for y in range(H):
                for x in range(W):
                    # Check Alpha: If pixel is transparent, skip
                    if C == 4 and img[y, x, 3] < 0.1:
                        continue

                    # Current Pixel
                    current_rgb = img[y, x, :3]
                    current_lum = luminance_map[y, x]

                    # Get Neighbors (3x3 window) from padded image
                    # Neighbors are at [y:y+3, x:x+3]
                    neighbors_window = img_padded[y : y + 3, x : x + 3]

                    # Flatten neighbors and remove the center pixel (which is at index 4 in 3x3)
                    neighbors_rgb = neighbors_window.reshape(
                        -1, neighbors_window.shape[-1]
                    )
                    neighbors_rgb = np.delete(neighbors_rgb, 4, axis=0)  # Remove center

                    # Filter valid neighbors (not transparent)
                    valid_neighbors = []
                    for n in neighbors_rgb:
                        # If RGBA, check alpha > 0.1, otherwise take it
                        if C == 3 or n[3] > 0.1:
                            valid_neighbors.append(n[:3])

                    if not valid_neighbors:
                        continue

                    valid_neighbors = np.array(valid_neighbors)

                    # Calculate Average Neighbor Color and Luminance
                    avg_neighbor_rgb = np.mean(valid_neighbors, axis=0)
                    avg_neighbor_lum = (
                        0.299 * avg_neighbor_rgb[0]
                        + 0.587 * avg_neighbor_rgb[1]
                        + 0.114 * avg_neighbor_rgb[2]
                    )

                    # --- CORE LOGIC ---
                    # Calculate Contrast Difference
                    # If neighbors are much brighter than current pixel -> It's a dark speckle
                    diff = avg_neighbor_lum - current_lum

                    if diff > contrast_threshold:
                        # This is a "bad" black/dark pixel.
                        # We want to make it look like the neighbors, but darker.

                        # Target Color = Neighbor Color * Darkening Factor (e.g., Dark Yellow)
                        target_rgb = avg_neighbor_rgb * darkening_factor

                        # Blend Original Black with Target Dark Color
                        # Strength 1.0 = Fully replaced by colored shadow
                        # Strength 0.0 = Keep original black
                        new_rgb = (target_rgb * soften_strength) + (
                            current_rgb * (1.0 - soften_strength)
                        )

                        out_img[y, x, :3] = new_rgb

            result_images.append(torch.from_numpy(out_img))

        return (torch.stack(result_images),)
