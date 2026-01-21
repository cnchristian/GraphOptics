from primatives import Block, BlockParam, TRAINABLE, NOT_TRAINABLE

import torch
from torch import Tensor
from torch.nn import Parameter

class PropagationBlock(Block):
    input_names = ("i",)
    output_names = ("o",)
    params = {
        "distance": BlockParam(Parameter(torch.tensor([torch.pi])), NOT_TRAINABLE),
    }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        distance = self.params["distance"].value
        return{
            "o": i * torch.exp(1j*distance)
        }

class MirrorBlock(Block):
    input_names = ("i1", "i2",)
    output_names = ("o1", "o2",)
    params = {
        "reflectance": BlockParam(Parameter(torch.tensor([0.5 + 0.0j])), NOT_TRAINABLE)
    }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i1 = inputs["i1"]
        i2 = inputs["i2"]
        R = self.params["reflectance"].value # TODO kinda want to be able to ignore the value part and just do math directly
        return {
            "o1": -i1*torch.sqrt(R) + i2*torch.sqrt(1-R),
            "o2": i1*torch.sqrt(1-R) + i2*torch.sqrt(R),
        }

