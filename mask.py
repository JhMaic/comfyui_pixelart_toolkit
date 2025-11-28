import cv2
import numpy as np
import torch


class Mask:
    CATEGORY = "💫PixelUtils/Mask"


class MaskHandleIsolatedPixels(Mask):
    DESCRIPTION = """
    A versatile tool for cleaning up pixel art masks.

    Modes:
    1. Area (Despeckle):
       Removes objects based on their pixel count (Area).
       - Uses 'area_threshold'.
       - Ignores 'search_distance_px'.
       - Good for: Removing static noise, single pixel dots.

    2. Proximity (Lonely):
       Removes objects based on how far they are from other objects.
       - Uses 'proximity_distance'.
       - Ignores 'max_island_size'.
       - Good for: Removing valid-looking shapes that are too far from the main character.

    Common Inputs:
    - operation: 'remove' (clean) or 'extract' (see what is being removed).
    """

    title = "Handle Isolated Pixels"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                # Switch between the two algorithms
                "filter_method": (
                    ["isolated_size", "isolated_distance"],
                    {"default": "isolated_size"},
                ),
                "operation": (["remove", "extract"],),
                # Parameter for "isolated_size" mode
                "max_island_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Used only in isolated_size mode. Max pixels to consider as noise.",
                    },
                ),
                # Parameter for "isolated_distance" mode
                "search_distance_px": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Used only in isolated_distance mode. Search radius in pixels.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    def execute(
        self, mask, filter_method, operation, max_island_size, search_distance_px
    ):
        # Dispatch to the correct function based on method
        if filter_method == "isolated_size":
            return self.process_area(mask, max_island_size, operation)
        else:
            return self.process_proximity(mask, search_distance_px, operation)

    def process_area(self, mask, threshold, operation):
        mask_np = mask.cpu().numpy()
        output_masks = []

        for m in mask_np:
            binary_map = (m > 0.5).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_map, connectivity=8
            )
            processed_mask = np.zeros_like(binary_map)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                is_target = area <= threshold

                if operation == "remove":
                    if not is_target:
                        processed_mask[labels == i] = 255
                elif operation == "extract":
                    if is_target:
                        processed_mask[labels == i] = 255

            output_masks.append(processed_mask.astype(np.float32) / 255.0)
        return (torch.from_numpy(np.array(output_masks)),)

    def process_proximity(self, mask, distance, operation):
        mask_np = mask.cpu().numpy()
        output_masks = []

        # Kernel for dilation
        k_size = distance * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        for m in mask_np:
            binary_map = (m > 0.5).astype(np.uint8) * 255
            H, W = binary_map.shape
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_map, connectivity=8
            )
            processed_mask = np.zeros_like(binary_map)

            for i in range(1, num_labels):
                # Optimization: ROI Crop
                x, y, w, h, area = stats[i]
                y1, y2 = max(0, y - distance), min(H, y + h + distance)
                x1, x2 = max(0, x - distance), min(W, x + w + distance)

                labels_roi = labels[y1:y2, x1:x2]
                current_obj_mask = (labels_roi == i).astype(np.uint8)
                other_objs_mask = (labels_roi > 0) & (labels_roi != i)

                dilated_current = cv2.dilate(current_obj_mask, kernel, iterations=1)
                has_neighbor = np.any(dilated_current & other_objs_mask)
                is_lonely = not has_neighbor

                if operation == "remove":
                    if not is_lonely:
                        processed_mask[labels == i] = 255
                elif operation == "extract":
                    if is_lonely:
                        processed_mask[labels == i] = 255

            output_masks.append(processed_mask.astype(np.float32) / 255.0)
        return (torch.from_numpy(np.array(output_masks)),)


class MaskApplyToImage(Mask):
    # -------------------------------------------------------------------
    # Node Description (English)
    # -------------------------------------------------------------------
    DESCRIPTION = """
    Applies a mask to an image by replacing the masked area with a specific RGBA color.

    Inputs:
    - image: The source image.
    - mask: The mask defining the area to replace.
    - fill_color_hex: The color to fill the masked area with (Hex format).
      - Format: #RRGGBB (Alpha=1.0) or #RRGGBBAA.
      - Default: #00000000 (Fully Transparent).
    - invert_mask:
      - False: The WHITE part of the mask becomes the 'fill_color'.
      - True: The BLACK part of the mask becomes the 'fill_color'.

    Output:
    - An RGBA image.
    """
    title = "Mask Apply To Image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "fill_color_hex": (
                    "STRING",
                    {"default": "#00000000", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"

    def apply_mask(self, image, mask, invert_mask, fill_color_hex):
        # 1. Parse Hex Color
        r, g, b, a = self.hex_to_rgba(fill_color_hex)

        # 2. Prepare Inputs
        # Convert Image to RGBA if it isn't already [Batch, H, W, 4]
        # Standard ComfyUI images are [Batch, H, W, 3] usually.
        if image.shape[-1] == 3:
            # Add an alpha channel of 1.0 (Opaque)
            alpha_channel = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2], 1), device=image.device
            )
            image_rgba = torch.cat((image, alpha_channel), dim=-1)
        else:
            image_rgba = image

        # 3. Resize Mask to match Image
        # Image: [B, H, W, C], Mask: [B, H, W]
        B, H, W, C = image_rgba.shape

        # Reshape mask for interpolation: [B, 1, H, W] (Channels first)
        mask_reshaped = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        # Interpolate to match image size
        mask_resized = torch.nn.functional.interpolate(
            mask_reshaped, size=(H, W), mode="bilinear", align_corners=False
        )

        # Reshape back to [B, H, W, 1] (Channels last) to match image layout
        mask_final = mask_resized.permute(0, 2, 3, 1)

        # 4. Handle Inversion
        # Logic: We want 'mask_weight' to be 1 where we want the COLOR, and 0 where we want the IMAGE.
        if invert_mask:
            mask_weight = 1.0 - mask_final
        else:
            mask_weight = mask_final

        # 5. Create Color Tensor
        # Create a solid color tensor of shape [1, 1, 1, 4]
        color_tensor = torch.tensor([r, g, b, a], device=image.device).view(1, 1, 1, 4)

        # 6. Composite (Linear Interpolation)
        # Formula: Result = Color * Mask + Original * (1 - Mask)
        # If Mask is 1 (White) -> Show Color
        # If Mask is 0 (Black) -> Show Original Image

        output_image = (color_tensor * mask_weight) + (image_rgba * (1.0 - mask_weight))

        return (output_image,)

    def hex_to_rgba(self, hex_code):
        """Converts hex string to (r, g, b, a) float tuple (0.0 - 1.0)"""
        hex_code = hex_code.strip().lstrip("#")

        try:
            if len(hex_code) == 6:
                # RGB -> RGBA (Alpha = 1.0)
                r = int(hex_code[0:2], 16) / 255.0
                g = int(hex_code[2:4], 16) / 255.0
                b = int(hex_code[4:6], 16) / 255.0
                return (r, g, b, 1.0)
            elif len(hex_code) == 8:
                # RGBA
                r = int(hex_code[0:2], 16) / 255.0
                g = int(hex_code[2:4], 16) / 255.0
                b = int(hex_code[4:6], 16) / 255.0
                a = int(hex_code[6:8], 16) / 255.0
                return (r, g, b, a)
            else:
                print(
                    f"Warning: Invalid hex code '{hex_code}'. Defaulting to transparent."
                )
                return (0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            print(f"Error parsing hex code: {e}. Defaulting to transparent.")
            return (0.0, 0.0, 0.0, 0.0)
