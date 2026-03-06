from primatives import Block, BlockParam, TRAINABLE, NOT_TRAINABLE, SuperBlock, Graph

import torch
from torch import Tensor
from torch.nn import Parameter

class GainBlock(Block):
    input_names = ("i",)
    output_names = ("o",)

    def __init__(self):
        super().__init__()
        self.params = {
            "factor": BlockParam(Parameter(torch.tensor([1.0 + 0.0j])), TRAINABLE),
        }

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        factor = self.params["factor"]
        return {
            "o": i * factor
        }

class SubtractBlock(Block):
    input_names = ("a","b",)
    output_names = ("c",)

    def __init__(self):
        super().__init__()
        self.params = {}

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        a = inputs["a"]
        b = inputs["b"]
        return {
            "c": a-b
        }

class SplitBlock(Block):
    input_names = ("i",)
    output_names = ("a","b")

    def __init__(self):
        super().__init__()
        self.params = {}

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        i = inputs["i"]
        return {
            "a": i,
            "b": i
        }

class FeedbackSuperBlock(SuperBlock):
    input_map = {
        "i": ("subtract", "a")
    }
    output_map = {
        "o": ("split", "a")
    }
    params = {}

    def generate_internal_graph(self) -> Graph:
        g = Graph()

        g.add_block("gain", GainBlock)
        g.add_block("subtract", SubtractBlock)
        g.add_block("split", SplitBlock)

        g.add_link("subtract", "c", "split", "i")
        g.add_link("split", "b", "gain", "i")
        g.add_link("gain", "o", "subtract", "b")

        return g