from dataclasses import dataclass

import networkx as nx

import torch
from torch.nn import Parameter, Module, ModuleDict
from torch import Tensor

TRAINABLE = True
NOT_TRAINABLE = False

# TODO trainability types for custom ranges and conditions on blockparam values
#  ideally this will also change the way blockparams are accessed to avoid the need to always ask for .value
class BlockParam:
    value: Parameter
    trainable: bool

    def __init__(self, value: Parameter, trainable: bool):
        self.value = value
        self.trainable = trainable

        self.value.requires_grad_(trainable)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = [arg.value if isinstance(arg, BlockParam) else arg for arg in args]
        return func(*new_args, **kwargs)

class Block(Module):
    input_names: tuple[str] = ()
    output_names: tuple[str] = ()

    def __init__(self):
        super().__init__()
        #self.params: dict[str, BlockParam] = {}

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

@dataclass(frozen=True)
class GraphIO:
    block_name: str
    block_port: str
    io_type: str

class GraphState:
    def __init__(self, graph):
        self.graph = graph
        self.values: list[Tensor] = []
        self.index: dict[GraphIO, int] = {}

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for io, index in self.index.items():
            yield io, self.values[index]

    def __contains__(self, key: GraphIO) -> bool:
        return key in self.index

    def __getitem__(self, key: GraphIO) -> Tensor:
        return self.values[self.index[key]]

    def __setitem__(self, key: GraphIO, value: Tensor):
        if key in self.index:
            self.values[self.index[key]] = value
        else:
            if any(value is item for item in self.values):
                index = next((i for i, item in enumerate(self.values) if item is value), None)
            else:
                index = len(self.values)
                self.values.append(value)
            self.index[key] = index

    def __add__(self, other): return torch.add(self, other)
    def __sub__(self, other): return torch.sub(self, other)
    def __mul__(self, other): return torch.mul(self, other)
    def __rmul__(self, other): return torch.mul(other, self)
    def __truediv__(self, other): return torch.div(self, other)
    def __neg__(self): return torch.neg(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        lengths = {len(arg) for arg in args if isinstance(arg, GraphState)}
        n = lengths.pop() if lengths else 0

        ref = next(arg for arg in args if isinstance(arg, GraphState))
        result = cls(ref.graph)
        result.index = ref.index.copy()

        for i in range(n):
            new_args = [arg.values[i] if isinstance(arg, GraphState) else arg for arg in args]
            result.values.append(func(*new_args, **kwargs))

        return result

    def flatten(self):
        return tuple(self.values)

    @classmethod
    def unflatten(cls, graph, flat, index):
        state = cls(graph)
        state.values = list(flat)
        state.index = index.copy()
        return state

    def reset(self, inputs: dict[str, Tensor]):
        for alias, value in inputs.items():
            alias_key = self.graph.inputs[alias]
            self[alias_key] = value

        for block_name, block in self.graph.blocks.items():
            for output_name in block.output_names:
                output_key = GraphIO(block_name=block_name, block_port=output_name, io_type="output")
                value = torch.tensor(0, dtype=torch.complex64, requires_grad=True)
                self[output_key] = value

        for link in self.graph.links:
            output_key = GraphIO(block_name=link.src_name, block_port=link.src_output, io_type="output")
            input_key = GraphIO(block_name=link.dst_name, block_port=link.dst_input, io_type="input")
            self[input_key] = self[output_key]

        for block_name, block in self.graph.blocks.items():
            for input_name in block.input_names:
                input_key = GraphIO(block_name=block_name, block_port=input_name, io_type="input")
                if not input_key in self:
                    value = torch.tensor(0, dtype=torch.complex64, requires_grad=True)
                    self[input_key] = value

from execution import execute
# TODO need to be able to handle moving a graph to the GPU (Currently everything starts on the CPU by default)
class Graph(Module):
    def __init__(self):
        super().__init__()

        self.blocks = ModuleDict()
        self.links: list[Link] = []
        self.nx = nx.DiGraph()

        self.inputs: dict[str, GraphIO] = {}
        self.outputs: dict[str, GraphIO] = {}

        self.state = GraphState(self)

    def add_block(self, name: str, block_type: type):
        if name in self.blocks:
            raise ValueError(f"Add block failed - Block with name \"{name}\" already exists")

        block = block_type()

        input_ports = " | ".join(f"<{i}>{i}" for i in block.input_names)
        output_ports = " | ".join(f"<{o}>{o}" for o in block.output_names)

        if input_ports and output_ports:
            label = f"{{ {{ {input_ports} }} | {name} | {{ {output_ports} }} }}"
        elif input_ports:
            label = f"{{ {{ {input_ports} }} | {name} }}"
        elif output_ports:
            label = f"{{ {name} | {{ {output_ports} }} }}"
        else:
            label = name

        self.blocks[name] = block
        self.nx.add_node(name, shape="record", label=label)

        block_params = block.params
        for param_name, data in block_params.items():
            self.register_parameter(f"{name}-{param_name}", data.value)

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
        self.nx.add_edge(src_name, dst_name, tailport=src_output, headport=dst_input)
        return link

    def write_param(self, name: str, param: str, data: BlockParam):
        if name not in self.blocks:
            raise KeyError(f"Write param failed - Block \"{name}\" does not exist")

        block = self.blocks[name]

        if param not in block.params:
            raise KeyError(f"Write param failed - Block \"{name}\" has no parameter \"{param}\"")

        block.params[param] = data
        self.register_parameter(f"{name}-{param}", data.value)

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

        computed_state = execute(self, inputs)

        output_dict = {}
        for alias, output_io in self.outputs.items():
            output_dict[alias] = computed_state[output_io]

        return output_dict

class SuperBlock(Block):
    input_map: dict[str, tuple[str, str]] = {}
    output_map: dict[str, tuple[str, str]] = {}

    def __init__(self):
        super().__init__()

        self.internal_graph = self.generate_internal_graph()

        self.input_names = (*self.input_map.keys(),)
        self.output_names = (*self.output_map.keys(),)

        for alias, (src_name, src_output) in self.output_map.items():
            self.internal_graph.set_output(alias, src_name, src_output)

        for alias, (dst_name, dst_input) in self.input_map.items():
            self.internal_graph.set_input(alias, dst_name, dst_input)

    def generate_internal_graph(self) -> Graph:
        raise NotImplementedError

    def compute(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.internal_graph.compute(**inputs)
