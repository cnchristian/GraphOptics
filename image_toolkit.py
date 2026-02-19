from primatives import Block, Packet, RealParam, ComplexParam, IntParam, EMPTY_VALUE, is_empty

import math
import torch
import torch.nn.functional as F

# ----------------------------------------------------- #
# ---------------------- Packets ---------------------- #
# ----------------------------------------------------- #

class ImagePacket(Packet):
    required_dim = 2
    required_params = {
        "height": IntParam,
        "width": IntParam,
        "min_value": RealParam,
        "max_value": RealParam,
    }

# ----------------------------------------------------- #
# ---------------------- Blocks ----------------------- #
# ----------------------------------------------------- #

class ValueScaleBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "min_value": RealParam(0),
            "max_value": RealParam(1),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        min_input = i["min_value"].val()
        max_input = i["max_value"].val()
        min_output = self.params["min_value"].val()
        max_output = self.params["max_value"].val()

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
            "padding": ComplexParam(0),
            "padded_width": IntParam(0),
            "padded_height": IntParam(0),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        width_input = i["width"].val()
        height_input = i["height"].val()
        width_output = self.params["padded_width"].val()
        height_output = self.params["padded_height"].val()

        if width_output < width_input or height_output < height_input:
            raise ValueError("PadBlock compute failed - negative padding requested")

        pad_top = math.ceil((height_output - height_input) / 2)
        pad_bottom = math.floor((height_output - height_input) / 2)
        pad_left = math.ceil((width_output - width_input) / 2)
        pad_right = math.floor((width_output - width_input) / 2)
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        data = {
            "height": self.params["padded_height"],
            "width": self.params["padded_width"],
            "min_value": i["min_value"],
            "max_value": i["max_value"],
        }

        return {
            "o": ImagePacket(data=data, value=torch.nn.functional.pad(image, padding))
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
            "cropped_width": IntParam(100),
            "cropped_height": IntParam(100),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        width_input = i["width"].val()
        height_input = i["height"].val()
        width_output = self.params["cropped_width"].val()
        height_output = self.params["cropped_height"].val()

        if width_output > width_input or height_output > height_input:
            raise ValueError("CropBlock compute failed - requested crop larger than input")

        crop_top = math.ceil((height_input - height_output) / 2)
        crop_bottom = crop_top + height_output

        crop_left = math.ceil((width_input - width_output) / 2)
        crop_right = crop_left + width_output

        data = {
            "height": self.params["cropped_height"],
            "width": self.params["cropped_width"],
            "min_value": i["min_value"],
            "max_value": i["max_value"],
        }

        return {
            "o": ImagePacket(data=data, value=image[:, crop_top:crop_bottom, crop_left:crop_right])
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
            "total_width": IntParam(100),
            "total_height": IntParam(100),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        width_input = i["width"].val()
        height_input = i["height"].val()
        width_output = self.params["total_width"].val()
        height_output = self.params["total_height"].val()

        if width_output < width_input or height_output < height_input:
            raise ValueError("TessellateBlock compute failed - output size smaller than input")

        # TODO understand why this y shift works only when different from x shift formula
        shift_y =  -((height_output // 2) % height_input)
        shift_x = ((width_output // 2) % width_input)*0
        image_centered = torch.roll(image, shifts=(shift_y, -shift_x), dims=(-2, -1))

        tiles_x = math.ceil(width_output / width_input) + 2
        tiles_y = math.ceil(height_output / height_input) + 2
        tiled = image_centered.repeat(1, tiles_y, tiles_x,)
        tiled_height = tiled.shape[-2]
        tiled_width = tiled.shape[-1]

        crop_top = (tiled_height - height_output) // 2
        crop_left = (tiled_width - width_output) // 2
        output = tiled[:, crop_top:crop_top + height_output, crop_left:crop_left + width_output]

        data = {
            "height": self.params["total_height"],
            "width": self.params["total_width"],
            "min_value": i["min_value"],
            "max_value": i["max_value"],
        }

        return {
            "o": ImagePacket(data=data, value=output)
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
            "individual_width": IntParam(100),
            "individual_height": IntParam(100),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value  # assume shape [H, W] or [C, H, W]

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        width_input = i["width"].val()
        height_input = i["height"].val()

        width_output_f = float(self.params["individual_width"].val())
        height_output_f = float(self.params["individual_height"].val())

        if width_output_f > width_input or height_output_f > height_input:
            raise ValueError("FragmentBlock compute failed - fragment size larger than input")

        # Final integer output size
        width_output = math.ceil(width_output_f)
        height_output = math.ceil(height_output_f)

        center_y = height_input / 2.0
        center_x = width_input / 2.0

        tiles_up = math.ceil(center_y / height_output_f)
        tiles_down = math.ceil((height_input - center_y) / height_output_f)
        tiles_left = math.ceil(center_x / width_output_f)
        tiles_right = math.ceil((width_input - center_x) / width_output_f)


        image = image.unsqueeze(1)
        B, _, H, W = image.shape

        fragments = []

        for ty in range(-tiles_up, tiles_down):
            for tx in range(-tiles_left, tiles_right):
                center_tile_y = center_y + ty * height_output_f
                center_tile_x = center_x + tx * width_output_f

                # Generate floating-point coordinate grid
                ys = torch.linspace(
                    center_tile_y - height_output_f / 2,
                    center_tile_y + height_output_f / 2,
                    height_output,
                    device=image.device,
                )

                xs = torch.linspace(
                    center_tile_x - width_output_f / 2,
                    center_tile_x + width_output_f / 2,
                    width_output,
                    device=image.device,
                )

                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

                # Normalize to [-1, 1] for grid_sample
                grid_y = (2.0 * grid_y / (H - 1)) - 1.0
                grid_x = (2.0 * grid_x / (W - 1)) - 1.0

                grid = torch.stack((grid_x, grid_y), dim=-1)
                grid = grid.repeat(B, 1, 1, 1)

                # Sample with bilinear interpolation
                fragment = F.grid_sample(
                    image,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )

                fragments.append(fragment)

        stacked = torch.stack(fragments, dim=0)
        averaged = stacked.mean(dim=0)

        # Remove channel dimension
        averaged = averaged.squeeze(1)

        data = {
            "height": height_output,
            "width": width_output,
            "min_value": i["min_value"],
            "max_value": i["max_value"],
        }

        return {
            "o": ImagePacket(data=data, value=averaged)
        }

class ScaleBlock(Block):
    inputs = {
        "i": ImagePacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "scale_factor": RealParam(1),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        image = i.value

        if is_empty(image):
            return {"o": ImagePacket(reference=i, value=EMPTY_VALUE)}

        scale = float(self.params["scale_factor"].val())

        data = {
            "height": IntParam(math.floor(i["height"].val() * scale)),
            "width": IntParam(math.floor(i["width"].val() * scale)),
            "min_value": i["min_value"],
            "max_value": i["max_value"],
        }

        return {
            "o": ImagePacket(data=data, value=F.interpolate(image.unsqueeze(1), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(1))
        }