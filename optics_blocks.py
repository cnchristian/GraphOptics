from primatives import Block, BlockParam, TRAINABLE, NOT_TRAINABLE

import torch
from torch import Tensor
from torch.nn import Parameter

class PropagationBlock(Block):
    input_names = ("i",)
    output_names = ("o",)

    def __init__(self):
        super().__init__()
        self.params = {
            "ds": BlockParam(Parameter(torch.tensor([9.2e-6])), NOT_TRAINABLE),
            "wavelength": BlockParam(Parameter(torch.tensor([561e-9])), NOT_TRAINABLE),
            "distance": BlockParam(Parameter(torch.tensor([1.0])), NOT_TRAINABLE)
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

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        distance = self.params["distance"].value
        wavelength = self.params["wavelength"].value
        ds = self.params["ds"].value.item()             # TODO need a better scheme for indicating that ds is fundamentally unlearnable, not just turned off

        if i.ndim > 2:
            raise ValueError("PropagationBlock compute failed - Inputs must be 2D, 1D, or scalar")
        i = i.view(1, -1) if i.dim() < 2 else i
        height, width = i.shape

        H = self.generate_H(distance, wavelength, width, height, ds)
        M = self.generate_M(width, height)

        return{
            "o": self.crop(torch.fft.ifft2(H * torch.fft.fft2(self.pad(i))) * M)
        }

class MirrorBlock(Block):
    input_names = ("i1", "i2",)
    output_names = ("o1", "o2",)

    def __init__(self):
        super().__init__()
        self.params = {
            "reflectance": BlockParam(Parameter(torch.tensor([0.5 + 0.0j])), NOT_TRAINABLE)
        }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i1 = inputs["i1"]
        i2 = inputs["i2"]
        R = self.params["reflectance"].value
        return {
            "o1": -i1*torch.sqrt(R) + i2*torch.sqrt(1-R),
            "o2": i1*torch.sqrt(1-R) + i2*torch.sqrt(R),
        }

class SLMBlock(Block):
    input_names = ("i", "phase")
    output_names = ("o",)

    def __init__(self):
        super().__init__()
        self.params = {
            "weights": BlockParam(Parameter(torch.tensor([1.0])), TRAINABLE),
            "biases": BlockParam(Parameter(torch.tensor([0.0])), TRAINABLE),
        }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        phase = inputs["phase"]
        W = self.params["weights"].value
        B = self.params["biases"].value

        return {
            "o": i * torch.exp(1j * (W*phase + B)),
        }

class LensBlock(Block):
    input_names = ("i",)
    output_names = ("o",)

    def __init__(self):
        super().__init__()
        self.params = {
            "ds": BlockParam(Parameter(torch.tensor([9.2e-6])), NOT_TRAINABLE),
            "wavelength": BlockParam(Parameter(torch.tensor([561e-9])), NOT_TRAINABLE),
            "focal_length": BlockParam(Parameter(torch.tensor([torch.inf])), TRAINABLE),
        }

    def generate_L(self, focal_length, wavelength, width, height, ds):
        f_x = torch.fft.fftfreq(width * 2, d=ds)
        f_y = torch.fft.fftfreq(height * 2, d=ds)
        F_Y, F_X = torch.meshgrid(f_y, f_x, indexing='ij')

        L = torch.exp(-1j * torch.pi * wavelength * focal_length * (F_X ** 2 + F_Y ** 2))

        return L

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        ds = self.params["ds"].value
        wavelength = self.params["wavelength"].value
        focal_length = self.params["focal_length"].value
        height, width = i.shape

        return {
            "o": i * self.generate_L(focal_length, wavelength, width, height, ds),
        }