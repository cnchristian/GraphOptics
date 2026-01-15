from primatives import Block, BlockParam, Graph
from primatives import TRAINABLE, NOT_TRAINABLE

import torch
from torch import Tensor

class PropagationBlock(Block):
    input_names = ("i",)
    output_names = ("o",)
    params = {
        "phase": BlockParam(torch.tensor([-1.0 + 0.0j]), NOT_TRAINABLE),
    }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        distance = self.params["phase"].value
        return{
            "o": i * distance
        }

class MirrorBlock(Block):
    input_names = ("i1", "i2",)
    output_names = ("o1", "o2",)
    params = {
        "reflectance": BlockParam(torch.tensor([0.5 + 0.0j]), NOT_TRAINABLE)
    }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i1 = inputs["i1"]
        i2 = inputs["i2"]
        R = self.params["reflectance"].value # TODO kinda want to be able to ignore the value part and just do math directly
        return {
            "o1": -i1*torch.sqrt(R) + i2*torch.sqrt(1-R),
            "o2": i1*torch.sqrt(1-R) + i2*torch.sqrt(R),
        }


g = Graph()

g.add_block("mirror_1", MirrorBlock)
g.add_block("mirror_2", MirrorBlock)

g.add_block("forward_prop", PropagationBlock)
g.add_block("backward_prop", PropagationBlock)

g.add_link("mirror_1", "o1", "forward_prop", "i")
g.add_link("forward_prop", "o", "mirror_2", "i2")
g.add_link("mirror_2", "o1", "backward_prop", "i")
g.add_link("backward_prop", "o", "mirror_1", "i2")

g.set_input("system_in", "mirror_1", "i1")
g.set_output("system_out", "mirror_2", "o2")

output = g.compute(system_in=torch.tensor(1, dtype=torch.complex64))
print(output)