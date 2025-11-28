class Converter:
    CATEGORY = "💫PixelUtils/Python Converter Wrapper"


class ToIntNode(Converter):
    title = "int(any, base)"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {}),
            },
            "optional": {
                "base": ("INT", {"min": 0, "default": 10, "display": "number"})
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, any, base):
        return (int(any, base),)


class ToFloatNode(Converter):
    title = "float(any)"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, any):
        return (float(any),)
