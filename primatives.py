from dataclasses import dataclass
import networkx as nx

from executor import GraphExecutor, GraphIO
from torch import Tensor

TRAINABLE = True
NOT_TRAINABLE = False

@dataclass
class BlockParam:
    value: Tensor
    trainable: bool

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        new_args = [arg.value if isinstance(arg, BlockParam) else arg for arg in args]
        return func(*new_args, **kwargs)

class Block:
    input_names: tuple[str] = ()
    output_names: tuple[str] = ()
    params: dict[str, BlockParam] = {}

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

class Link:
    def __init__(self, src_name, src_output, dst_name, dst_input):
        self.src_name = src_name
        self.src_output = src_output
        self.dst_name = dst_name
        self.dst_input = dst_input

    def __repr__(self):
        return f"{self.src_name}.{self.src_output} → {self.dst_name}.{self.dst_input}"

# TODO need to be able to handle moving a graph to the GPU (Currently everything starts on the CPU by default)
# TODO need to decide how states will be handled (i.e. should they still belong to the graphexecutor or should they be a property of the graph, or something else entirely?)
class Graph:
    def __init__(self):
        self.blocks: dict[str, Block] = {}
        self.links: list[Link] = []
        self.nx = nx.DiGraph()
        self.inputs: dict[str, GraphIO] = {}
        self.outputs: dict[str, GraphIO] = {}
        self.executor = GraphExecutor(self)

    def __repr__(self):
        s = "Graph:\n"
        for name, block in self.blocks:
            s += f"  Block {name}: inputs={block.input_names}, outputs={block.output_names}\n"

        s += "  Links:\n"
        for l in self.links:
            s += f"    {l}\n"
        return s

    def add_block(self, name: str, block_type: type):
        if name in self.blocks:
            raise ValueError(f"Add block failed - Block with name \"{name}\" already exists")

        block = block_type()

        self.blocks[name] = block
        self.nx.add_node(name)
        return block

    def add_link(self, src_name: str, src_output: str,
                 dst_name: str, dst_input: str):
        if src_name not in self.blocks:
            raise KeyError(f"Add link failed - Block with name \"{src_name}\" does not exist")
        if src_output not in self.blocks[src_name].output_names:
            raise ValueError(f"Add link failed - Block \"{src_name}\" does not have output \"{src_output}\"")
        if dst_name not in self.blocks:
            raise KeyError(f"Add link failed - Block with name \"{dst_name}\" does not exist")
        if dst_input not in self.blocks[dst_name].input_names:
            raise ValueError(f"Add link failed - Block \"{dst_name}\" does not have input \"{dst_input}\"")

        link = Link(src_name, src_output, dst_name, dst_input)
        self.links.append(link)
        self.nx.add_edge(src_name, dst_name)
        return link

    def write_param(self, name: str, param: str, data: BlockParam):
        if name not in self.blocks:
            raise KeyError(f"Write param failed - Block \"{name}\" does not exist")

        block = self.blocks[name]

        if param not in block.params:
            raise KeyError(f"Write param failed - Block \"{name}\" has no parameter \"{param}\"")

        block.params[param] = data

    def set_input(self, alias: str, dst_name: str, dst_input: str):
        if alias in self.inputs:
            raise ValueError(f"Set input failed - alias \"{alias}\" already exists")
        if dst_name not in self.blocks:
            raise KeyError(f"Set input failed - Block \"{dst_name}\" does not exist")
        if dst_input not in self.blocks[dst_name].input_names:
            raise KeyError(f"Set input failed - Block \"{dst_name}\" does not have input \"{dst_input}\"")

        input_io = GraphIO(dst_name, dst_input, "input")

        if input_io in self.inputs.values():
            raise ValueError(f"Set input param failed - input \"{dst_input}\" of block \"{dst_name}\" "
                             f"has already been assigned an alias")

        self.inputs[alias] = input_io
        return input_io

    def set_output(self, alias: str, src_name: str, src_output: str):
        if alias in self.inputs:
            raise ValueError(f"Set output failed - alias \"{alias}\" already exists")
        if src_name not in self.blocks:
            raise KeyError(f"Set output failed - Block \"{src_name}\" does not exist")
        if src_output not in self.blocks[src_name].output_names:
            raise KeyError(f"Set output failed - Block \"{src_name}\" does not have output \"{src_output}\"")

        output_io = GraphIO(src_name, src_output, "output")

        if output_io in self.outputs.values():
            raise ValueError(f"Set input param failed - output \"{src_output}\" of block \"{src_name}\" "
                             f"has already been assigned an alias")

        self.outputs[alias] = output_io
        return output_io

    def compute(self, **inputs):
        if not inputs.keys() == self.inputs.keys():
            raise KeyError("Graph compute failed - incorrect inputs provided")

        computed_state = self.executor.execute(inputs)

        output_dict = {}
        for alias, output_io in self.outputs.items():
            output_dict[alias] = computed_state[output_io]

        return output_dict

class SuperBlock(Block):
    input_map: dict[str, tuple[str, str]] = {}
    output_map: dict[str, tuple[str, str]] = {}

    def __init__(self):
        self.internal_graph = self.generate_internal_graph()

        for alias, (src_name, src_output) in self.output_map.items():
            self.internal_graph.set_output(alias, src_name, src_output)

        for alias, (dst_name, dst_input) in self.input_map.items():
            self.internal_graph.set_input(alias, dst_name, dst_input)

    def generate_internal_graph(self) -> Graph:
        raise NotImplementedError

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.internal_graph.compute(**inputs)
