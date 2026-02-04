from primatives import Block, BlockParam, Packet, TRAINABLE, NOT_TRAINABLE

import math
import torch
from torch.nn import Parameter

# ----------------------------------------------------- #
# ---------------------- Packets ---------------------- #
# ----------------------------------------------------- #

class ImagePacket(Packet):
    required_keys = {"height", "width", "min", "max"}

    def default_value(self):
        h, w = self.data["height"], self.data["width"]
        return torch.ones(h, w, dtype=torch.complex64) * self.data["min"]

# ----------------------------------------------------- #
# ---------------------- Blocks ----------------------- #
# ----------------------------------------------------- #

class RescaleBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "min_value": BlockParam(Parameter(torch.tensor(0)), NOT_TRAINABLE),
            "max_value": BlockParam(Parameter(torch.tensor(1)), NOT_TRAINABLE),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value
        min_input = i["min_value"].value
        max_input = i["max_value"].value
        min_output = self.params["min_value"].value
        max_output = self.params["max_value"].value

        scale = (max_output - min_output) / (max_input - min_input)

        return {
            "o": ImagePacket(reference=i, value=(image-min_input)*scale+min_output)
        }

class PadBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "padding": BlockParam(Parameter(torch.tensor(0)), NOT_TRAINABLE),
            "padded_width": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
            "padded_height": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        width_input = i["width"].value
        height_input = i["height"].value
        width_output = self.params["padded_width"].value
        height_output = self.params["padded_height"].value

        if width_output < width_input or height_output < height_input:
            raise ValueError("PadBlock compute failed - negative padding requested")

        pad_top = math.ceil((height_output - height_input) / 2)
        pad_bottom = math.floor((width_output - width_input) / 2)
        pad_left = math.ceil((width_output - width_input) / 2)
        pad_right = math.floor((width_output - width_input) / 2)
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        return {
            "o": ImagePacket(reference=i, value=torch.nn.functional.pad(image, padding))
        }

class CropBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "cropped_width": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
            "cropped_height": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        width_input = i["width"].value
        height_input = i["height"].value
        width_output = self.params["cropped_width"].value
        height_output = self.params["cropped_height"].value

        if width_output > width_input or height_output > height_input:
            raise ValueError("CropBlock compute failed - requested crop larger than input")

        crop_top = math.ceil((height_input - height_output) / 2)
        crop_bottom = crop_top + height_output

        crop_left = math.ceil((width_input - width_output) / 2)
        crop_right = crop_left + width_output

        return {
            "o": ImagePacket(reference=i, value=image[crop_top:crop_bottom, crop_left:crop_right])
        }

class TessellateBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "total_width": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
            "total_height": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        width_input = i["width"].value
        height_input = i["height"].value
        width_output = self.params["total_width"].value
        height_output = self.params["total_height"].value

        if width_output < width_input or height_output < height_input:
            raise ValueError("TessellateBlock compute failed - output size smaller than input")

        tiles_x = math.ceil(width_output / width_input) + 2
        tiles_y = math.ceil(height_output / height_input) + 2
        tiled = image.repeat(tiles_y, tiles_x,)
        tiled_height = tiled.shape[-2]
        tiled_width = tiled.shape[-1]

        crop_top = (tiled_height - height_output) // 2
        crop_left = (tiled_width - width_output) // 2
        output = tiled[crop_top:crop_top + height_output, crop_left:crop_left + width_output]

        return {
            "o": ImagePacket(reference=i, value=output)
        }

class FragmentBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "individual_width": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
            "individual_height": BlockParam(Parameter(torch.tensor(100)), NOT_TRAINABLE),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        width_input = i["width"].value
        height_input = i["height"].value
        width_output = self.params["individual_width"].value
        height_output = self.params["individual_height"].value

        if width_output > width_input or height_output > height_input:
            raise ValueError("FragmentBlock compute failed - fragment size larger than input")

        center_y = height_input // 2
        center_x = width_input // 2

        tiles_up = math.ceil(center_y / height_output)
        tiles_down = math.ceil((height_input - center_y) / height_output)
        tiles_left = math.ceil(center_x / width_output)
        tiles_right = math.ceil((width_input - center_x) / width_output)

        fragments = []
        for ty in range(-tiles_up, tiles_down):
            for tx in range(-tiles_left, tiles_right):
                top = center_y + ty * height_output - height_output // 2
                left = center_x + tx * width_output - width_output // 2
                bottom = top + height_output
                right = left + width_output
                if top < 0 or left < 0 or bottom > height_input or right > width_input:
                    continue

                fragment = image[top:bottom, left:right]
                fragments.append(fragment)

        stacked = torch.stack(fragments, dim=0)
        averaged = stacked.mean(dim=0)

        return {
            "o": ImagePacket(reference=i, value=averaged)
        }