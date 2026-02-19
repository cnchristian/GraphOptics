from primatives import Block, Packet, RealParam, IntParam, is_empty, EMPTY_VALUE, SuperBlock, Graph
from image_toolkit import ImagePacket, CropBlock, ScaleBlock

import torch

# ----------------------------------------------------- #
# ---------------------- Packets ---------------------- #
# ----------------------------------------------------- #

class FieldPacket(Packet):
    required_dim = 2
    required_params = {
        "height": IntParam,
        "width": IntParam,
        "ds": RealParam,
        "wavelength": RealParam,
    }

# ----------------------------------------------------- #
# ---------------------- Blocks ----------------------- #
# ----------------------------------------------------- #

class PropagationBlock(Block):
    inputs = {
        "i": FieldPacket,
    }
    outputs = {
        "o": FieldPacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "distance": RealParam(1)
        }

    def generate_H(self, distance, wavelength, width, height, ds):
        f_x = torch.fft.fftfreq(width * 2, d=ds)
        f_y = torch.fft.fftfreq(height * 2, d=ds)
        F_Y, F_X = torch.meshgrid(f_y, f_x, indexing='ij')

        du_x = 1 / (width * 2 * ds)
        du_y = 1 / (height * 2 * ds)
        u_limit_x = 1 / (torch.sqrt((2 * du_x * torch.abs(distance)) ** 2 + 1) * wavelength)
        u_limit_y = 1 / (torch.sqrt((2 * du_y * torch.abs(distance)) ** 2 + 1) * wavelength)

        H_limit = (((F_Y ** 2 / u_limit_y ** 2 + F_X ** 2 * wavelength ** 2) < 1) * (
                (F_X ** 2 / u_limit_x ** 2 + F_Y ** 2 * wavelength ** 2) < 1))

        H = torch.exp(1j * 2 * torch.pi / wavelength * distance *
                      torch.sqrt(1 - (wavelength * F_X) ** 2 - (wavelength * F_Y) ** 2)) * H_limit
        return H.unsqueeze(0)

    def generate_M(self, width, height):
        offset_w = width // 2
        offset_h = height // 2
        start_w, end_w = offset_w, offset_w + width
        start_h, end_h = offset_h, offset_h + height

        # Create a 2D mask for the 28x28 image
        M = torch.zeros((2*height, 2*width), dtype=torch.complex64)
        M[start_h:end_h, start_w:end_w] = 1.0

        return M.unsqueeze(0)

    def pad(self, i):
        B, H, W = i.shape
        h_start = H // 2
        w_start = W // 2

        i_pad = torch.zeros((B, 2 * H, 2 * W), dtype=torch.complex64)
        i_pad[:, h_start:h_start + H, w_start:w_start + W] = i

        return i_pad

    def crop(self, o):
        B, H, W = o.shape
        h_half = H // 4
        w_half = W // 4

        return o[:, h_half:h_half + H // 2, w_half:w_half + W // 2]

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field = i.value

        if is_empty(field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        height = i["height"].val()
        width = i["width"].val()
        ds = i["ds"].val()
        wavelength = i["wavelength"].val()

        distance = self.params["distance"].val()

        H = self.generate_H(distance, wavelength, width, height, ds)
        M = self.generate_M(width, height)

        return{
            "o": FieldPacket(reference=i, value=self.crop(torch.fft.ifft2(H * torch.fft.fft2(self.pad(field))) * M))
        }

class MirrorBlock(Block):
    inputs = {
        "i1": FieldPacket,
        "i2": FieldPacket,
    }
    outputs = {
        "o1": FieldPacket,
        "o2": FieldPacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "reflectance": RealParam(0.5)
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i1 = inputs["i1"]
        i2 = inputs["i2"]

        h1, w1, h2, w2 = i1["height"].val(), i1["width"].val(), i2["height"].val(), i2["width"].val()

        field1 = i1.value
        field2 = i2.value

        B = field1.shape[0]
        if is_empty(field1) and is_empty(field2):
            return {"o1": FieldPacket(reference=i1, value=EMPTY_VALUE),
                    "o2": FieldPacket(reference=i2, value=EMPTY_VALUE)}
        elif is_empty(field1):
            ref = i2
            field1 = torch.zeros((B, h2, w2), dtype=torch.complex64)
        elif is_empty(field2):
            ref = i1
            field2 = torch.zeros((B, h1, w1), dtype=torch.complex64)
        else:
            ref = i1
            if (h1, w1) != (h2, w2):
                raise ValueError("MirrorBlock compute error - incoming fields do not have same shape")

        R = self.params["reflectance"].val()

        return {
            "o1": FieldPacket(reference=ref, value=-field1*torch.sqrt(R) + field2*torch.sqrt(1-R)),
            "o2": FieldPacket(reference=ref, value=field1*torch.sqrt(1-R) + field2*torch.sqrt(R)),
        }

class SLMBlock(Block):
    inputs = {
        "i": FieldPacket,
        "phase": ImagePacket,
    }
    outputs = {
        "o": FieldPacket,
    }
    # TODO need to have the requirements written here somehow
    #  and some check to make sure the user has specified them before computing

    def __init__(self):
        super().__init__()
        self.params = {
            "weights": RealParam(1),
            "biases": RealParam(0),
            "undiffracted": RealParam(0.5)
        }

    def refresh(self):
        h, w = int(self.requirements["height"]), int(self.requirements["width"])
        # TODO need to adjust this so that I dont have to rewrite undiffracted
        self.params = {
            "weights": RealParam(torch.ones(h, w)),
            "biases": RealParam(torch.zeros(h, w)),
            "undiffracted": RealParam(self.params["undiffracted"].val())
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        i_field = i.value
        phase = inputs["phase"]
        phase_field = phase.value

        h, w = int(self.requirements["height"]), int(self.requirements["width"])

        B = phase_field.shape[0]
        if is_empty(i_field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}
        elif is_empty(phase_field):
            phase_field = torch.zeros((B, h, w), dtype=torch.complex64)

        if not (i_field.shape == phase_field.shape == (B, h, w)):
            raise ValueError(f"SLMBlock compute error - phase array and input do not have matching required shape (B, {h}, {w})")

        W = self.params["weights"].val()
        B = self.params["biases"].val()
        u = self.params["undiffracted"].val()

        return {
            "o": FieldPacket(reference=i, value=u*i_field + (1-u)*(i_field * torch.exp(1j * (W*phase_field + B)))),
        }

class LensBlock(Block):
    inputs = {
        "i": FieldPacket,
    }
    outputs = {
        "o": FieldPacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "focal_length": RealParam(torch.inf),
        }

    def generate_L(self, focal_length, wavelength, width, height, ds):
        x = (torch.arange(width) - width // 2) * ds
        y = (torch.arange(height) - height // 2) * ds
        Y, X = torch.meshgrid(y, x, indexing='ij')

        L = torch.exp(-1j * torch.pi / (wavelength * focal_length) * (X ** 2 + Y ** 2))

        return L.unsqueeze(0)

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field = i.value

        if is_empty(field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        height = i["height"].val()
        width = i["width"].val()
        ds = i["ds"].val()
        wavelength = i["wavelength"].val()
        focal_length = self.params["focal_length"].val()

        return {
            "o": FieldPacket(reference=i, value=field * self.generate_L(focal_length, wavelength, width, height, ds)),
        }

class DetectorBlock(Block):
    inputs = {
        "i": FieldPacket,
    }
    outputs = {
        "o": ImagePacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "max_value": IntParam(255)
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field = i.value
        max_value = self.params["max_value"]

        if is_empty(field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        data = {
            "height": i["height"],
            "width": i["width"],
            "min_value": IntParam(0),
            "max_value": max_value,
        }
        raw_o = torch.pow(torch.abs(field), 2)

        return {
            "o": ImagePacket(data=data, value=raw_o/torch.max(raw_o)*max_value.val()),
        }

class CameraBlock(SuperBlock):
    input_map = {
        "i": ("detector", "i"),
    }
    output_map = {
        "o": ("crop", "o"),
    }

    def __init__(self):
        self.params = {
            "camera_width": IntParam(1440),
            "camera_height": IntParam(1080),
            "camera_ds": RealParam(3.45e-6)
        }
        super().__init__()

    def generate_internal_graph(self) -> Graph:
        g = Graph()

        g.add_block("detector", DetectorBlock)
        g.add_block("scale", ScaleBlock)
        g.add_block("crop", CropBlock)

        g.add_link("detector", "o", "scale", "i")
        g.add_link("scale", "o", "crop", "i")

        return g

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        i_ds = i["ds"].val()
        c_ds = self.params["camera_ds"].val()

        scale = RealParam(float(i_ds/c_ds))
        self.internal_graph.write_params("scale", scale_factor=scale)
        self.internal_graph.write_params("crop", cropped_width=self.params["camera_width"], cropped_height=self.params["camera_height"])

        return super().compute(inputs)

class FourFBlock(Block):
    inputs = {
        "i": FieldPacket,
    }
    outputs = {
        "o": FieldPacket,
    }

    def __init__(self):
        super().__init__()
        self.params = {
            "f1": RealParam(1),
            "f2": RealParam(1),
            "mask": RealParam(1),
        }

    def refresh(self):
        h, w = int(self.requirements["height"]), int(self.requirements["width"])
        self.params = {
            "f1": RealParam(1),
            "f2": RealParam(1),
            "mask": RealParam(torch.ones(h, w)),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field_I = i.value

        if is_empty(field_I):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        f1 = self.params["f1"].val()
        f2 = self.params["f2"].val()
        mask = self.params["mask"].val()

        field_F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field_I), norm="ortho"))
        field_M = field_F * mask
        field_O = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field_M), norm="ortho"))

        data = {
            "height": i["height"],
            "width": i["width"],
            "ds": RealParam(i["ds"].val() * f2 / f1),
            "wavelength": i["wavelength"],
        }

        return {
            "o": FieldPacket(data=data, value=field_O),
        }

# TODO Finish this
class ASMFourFBlock(SuperBlock):
    input_map = {
        "i": ("detector", "i"),
    }
    output_map = {
        "o": ("crop", "o"),
    }

    def __init__(self):
        self.params = {
            "camera_width": IntParam(1440),
            "camera_height": IntParam(1080),
            "camera_ds": RealParam(3.45e-6)
        }
        super().__init__()

    def generate_internal_graph(self) -> Graph:
        g = Graph()

        g.add_block("detector", DetectorBlock)
        g.add_block("scale", ScaleBlock)
        g.add_block("crop", CropBlock)

        g.add_link("detector", "o", "scale", "i")
        g.add_link("scale", "o", "crop", "i")

        # g.add_block("L1", LensBlock)
        # g.add_block("L1_prop_1_forward", PropagationBlock)
        # g.add_block("L1_prop_2_forward", PropagationBlock)
        # g.write_params("L1_prop_1_forward", distance=l1_focal_length)
        # g.write_params("L1", focal_length=l1_focal_length)
        # g.write_params("L1_prop_2_forward", distance=l1_focal_length)

        # g.add_block("L2", LensBlock)
        # g.add_block("L2_prop_1_forward", PropagationBlock)
        # g.add_block("L2_prop_2_forward", PropagationBlock)
        # g.write_params("L2_prop_1_forward", distance=l2_focal_length)
        # g.write_params("L2", focal_length=l2_focal_length)
        # g.write_params("L2_prop_2_forward", distance=l2_focal_length)

        # g.add_link("mirror", "o1", "L1_prop_1_forward", "i")
        # g.add_link("L1_prop_1_forward", "o", "L1", "i")
        # g.add_link("L1", "o", "L1_prop_2_forward", "i")
        # g.add_link("L1_prop_2_forward", "o", "L2_prop_1_forward", "i")
        # g.add_link("L2_prop_1_forward", "o", "L2", "i")
        # g.add_link("L2", "o", "L2_prop_2_forward", "i")

        return g

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        i_ds = i["ds"].val()
        c_ds = self.params["camera_ds"].val()

        scale = RealParam(float(i_ds / c_ds))
        self.internal_graph.write_params("scale", scale_factor=scale)
        self.internal_graph.write_params("crop", cropped_width=self.params["camera_width"],
                                         cropped_height=self.params["camera_height"])

        return super().compute(inputs)