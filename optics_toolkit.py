from primatives import Block, Packet, RealParam, IntParam, is_empty, EMPTY_VALUE
from image_toolkit import ImagePacket

import torch

# ----------------------------------------------------- #
# ---------------------- Packets ---------------------- #
# ----------------------------------------------------- #

class FieldPacket(Packet):
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
        return H

    def generate_M(self, width, height):
        offset_w = width // 2
        offset_h = height // 2
        start_w, end_w = offset_w, offset_w + width
        start_h, end_h = offset_h, offset_h + height

        # Create a 2D mask for the 28x28 image
        M = torch.zeros((2*height, 2*width), dtype=torch.complex64)
        M[start_h:end_h, start_w:end_w] = 1.0

        return M

    def pad(self, i):
        H, W = i.shape
        h_start = H // 2
        w_start = W // 2

        i_pad = torch.zeros((2 * H, 2 * W), dtype=torch.complex64)
        i_pad[h_start:h_start + H, w_start:w_start + W] = i

        return i_pad

    def crop(self, o):
        H, W = o.shape
        h_half = H // 4
        w_half = W // 4

        return o[h_half:h_half + H // 2, w_half:w_half + W // 2]

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field = i.value

        if is_empty(field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        height = i["height"].value
        width = i["width"].value
        ds = i["ds"].value
        wavelength = i["wavelength"].value

        distance = self.params["distance"].value

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

        if is_empty(field1) and is_empty(field2):
            return {"o1": FieldPacket(reference=i1, value=EMPTY_VALUE),
                    "o2": FieldPacket(reference=i2, value=EMPTY_VALUE)}
        elif is_empty(field1):
            ref = i2
            field1 = torch.zeros((h2, w2), dtype=torch.complex64)
        elif is_empty(field2):
            ref = i1
            field2 = torch.zeros((h1, w1), dtype=torch.complex64)
        else:
            ref = i1
            if (h1, w1) != (h2, w2):
                raise ValueError("MirrorBlock compute error - incoming fields do not have same shape")

        R = self.params["reflectance"].value

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

    def __init__(self):
        super().__init__()
        self.params = {
            "weights": RealParam(1),
            "biases": RealParam(0),
        }

    def refresh(self):
        h, w = int(self.requirements["height"]), int(self.requirements["width"])
        self.params = {
            "weights": RealParam(torch.ones(h, w)),
            "biases": RealParam(torch.zeros(h, w)),
        }

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        i_field = i.value
        phase = inputs["phase"]
        phase_field = phase.value

        h, w = int(self.requirements["height"]), int(self.requirements["width"])

        if is_empty(i_field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}
        elif is_empty(phase_field):
            phase_field = torch.zeros((h, w), dtype=torch.complex64)

        if not (i_field.shape == phase_field.shape == (h, w)):
            raise ValueError(f"SLMBlock compute error - phase array or input does not have required shape ({h}, {w})")

        W = self.params["weights"].value
        B = self.params["biases"].value

        return {
            "o": FieldPacket(reference=i, value=i_field * torch.exp(1j * (W*phase_field + B))),
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
        f_x = torch.fft.fftfreq(width * 2, d=ds)
        f_y = torch.fft.fftfreq(height * 2, d=ds)
        F_Y, F_X = torch.meshgrid(f_y, f_x, indexing='ij')

        L = torch.exp(-1j * torch.pi * wavelength * focal_length * (F_X ** 2 + F_Y ** 2))

        return L

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        i = inputs["i"]
        field = i.value

        if is_empty(field):
            return {"o": FieldPacket(reference=i, value=EMPTY_VALUE)}

        height = i["height"].value
        width = i["width"].value
        ds = i["ds"].value
        wavelength = i["wavelength"].value
        focal_length = self.params["focal_length"].value

        return {
            "o": FieldPacket(reference=i, value=field * self.generate_L(focal_length, wavelength, width, height, ds)),
        }