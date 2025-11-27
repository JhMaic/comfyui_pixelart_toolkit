# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
from .color import (
    PickPixelColorAtPosition,
    PixelArtReduceColors,
    ColorConverter,
    PixelArtShadowSoftener,
)
from .converters import ToInt, ToFloat
from .mask import MaskHandleIsolatedPixels, MaskApplyToImage

NODE_CLASS_MAPPINGS = {
    "ToInt": ToInt,
    "ToFloat": ToFloat,
    "COLOR_PickColor": PickPixelColorAtPosition,
    "COLOR_ReduceColor": PixelArtReduceColors,
    "COLOR_Converter": ColorConverter,
    "COLOR_PixelArtShadowSoftener": PixelArtShadowSoftener,
    "MASK_HandleIsolatedPixels": MaskHandleIsolatedPixels,
    "MASK_AppleToImage": MaskApplyToImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ToInt": ToInt.title,
    "ToFloat": ToFloat.title,
    "COLOR_PickColor": PickPixelColorAtPosition.title,
    "COLOR_ReduceColor": PixelArtReduceColors.title,
    "COLOR_Converter": ColorConverter.title,
    "COLOR_PixelArtShadowSoftener": PixelArtShadowSoftener.title,
    "MASK_HandleIsolatedPixels": MaskHandleIsolatedPixels.title,
    "MASK_AppleToImage": MaskApplyToImage.title,
}
