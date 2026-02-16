from abc import abstractmethod
from dataclasses import dataclass

import networkx as nx

import torch
from torch import Tensor
from torch.nn import Parameter, Module, ModuleDict

EMPTY_VALUE = torch.empty(0)
def is_empty(value: Tensor) -> bool:
    return value.numel() == 0

class Param:
    trainability = None

    @abstractmethod
    def __init__(self, value=None):
        if value is None:
            value = self.default_val()
        self._init(value)

    def _init(self, value):
        self.value = Parameter(value) if self.trainability else value

    def default_val(self):
        return 0

    def val(self):
        raise NotImplementedError

class IntParam(Param):
    trainability = False

    def _init(self, value=None):
        if not isinstance(value, int):
            raise TypeError("IntParam create failed - value must be integer")
        super()._init(value)

    def val(self):
        return int(self.value)

class TensorParam(Param):
    trainability = True

    @abstractmethod
    def _init(self, value=None):
        super()._init(value)
        self.trainable = False
        self.value.requires_grad_(False)

    def val(self):
        return self.value

    def set_trainable(self, trainable):
        self.trainable = trainable
        self.value.requires_grad_(trainable)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = [arg.value if isinstance(arg, cls) else arg for arg in args]
        return func(*new_args, **kwargs)

class RealParam(TensorParam):
    def _init(self, value=None):
        value = torch.tensor(value, dtype=torch.float32)
        super()._init(value)

class ComplexParam(TensorParam):
    def _init(self, value=None):
        value = torch.tensor(value, dtype=torch.complex64)
        super()._init(value)

class Packet:
    required_params: dict[str, type] = {}

    @abstractmethod
    def __init__(self, reference = None, data = None, value = None):
        if reference is not None and data is None:
            if type(reference) is not type(self):
                raise TypeError(f"Create packet failed - incorrect reference type {type(reference)}")
            self.data = reference.data
        elif reference is None and data is not None:
            if data.keys() != self.required_params.keys():
                raise KeyError(f"Create packet failed - incorrect parameters provided")
            self.data = data
        elif reference is not None and data is not None:
            raise ValueError(f"{type(self)} creation failed - conflicting reference and parameters")
        else:
            self.data = {key: cls() for key, cls in self.required_params.items()}

        if value is None:
            value = EMPTY_VALUE
        self.value = value

    def __getitem__(self, key: str) -> Param:
        return self.data[key]

    def set_data(self, data: dict[str, Param]):
        if self.data.keys() != data.keys():
            raise KeyError("Set packet data failed - incorrect format")

        self.data = data

class Block(Module):
    inputs: dict[str, type] = {}
    outputs: dict[str, type] = {}
    requirements: dict[str, Tensor] = {}

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.input_names = tuple(self.inputs.keys())
        self.output_names = tuple(self.outputs.keys())

    def refresh(self):
        raise NotImplementedError

    # TODO need to come up with a better paradigm for understanding how to handle empty inputs automatically
    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        raise NotImplementedError

class Link:
    def __init__(self, src_name, src_output, dst_name, dst_input, packet_type):
        self.src_name = src_name
        self.src_output = src_output
        self.dst_name = dst_name
        self.dst_input = dst_input
        self.packet_type = packet_type

@dataclass(frozen=True)
class GraphIO:
    block_name: str
    block_port: str
    io_type: str

class GraphState:
    def __init__(self, graph):
        self.graph = graph
        self.packets: list[Packet] = []
        self.index: dict[GraphIO, int] = {}

    def __len__(self):
        return len(self.packets)

    def __iter__(self):
        for io, index in self.index.items():
            yield io, self.packets[index]

    def __contains__(self, key: GraphIO) -> bool:
        return key in self.index

    def __getitem__(self, key: GraphIO) -> Packet:
        return self.packets[self.index[key]]

    def __setitem__(self, key: GraphIO, packet: Packet):
        if key in self.index:
            self.packets[self.index[key]] = packet
        else:
            if any(packet is item for item in self.packets):
                index = next((i for i, item in enumerate(self.packets) if item is packet), None)
            else:
                index = len(self.packets)
                self.packets.append(packet)
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

        # TODO this has become so convoluted that I should really just hardcode the few functions that are necessary
        for i in range(n):
            new_args = [arg.packets[i] if isinstance(arg, GraphState) else arg for arg in args]
            idx = next((i for i, arg in enumerate(new_args) if not is_empty(arg.value)), 0)
            if any(not is_empty(a.value) for a in new_args):
                for a in new_args:
                    a.value = a.value if not is_empty(a.value) else torch.tensor(0, dtype=torch.complex64)
            result.packets.append(type(new_args[idx])(reference=new_args[idx], value=func(*[new_arg.value for new_arg in new_args], **kwargs)))
            for a in new_args:
                if a.value.numel() == 1 and a.value == torch.tensor([0], dtype=torch.complex64):
                    a.value = EMPTY_VALUE

        return result

    def flatten(self):
        return self.packets, tuple([packet.value for packet in self.packets])

    @classmethod
    def unflatten(cls, graph, packets, flat, index):
        state = cls(graph)
        state.packets = packets.copy()
        state.index = index.copy()
        for i, packet in enumerate(state.packets):
            packet.value = flat[i]
        return state

    def reset(self, inputs: dict[str, Packet]):
        for alias, packet in inputs.items():
            alias_key = self.graph.inputs[alias]
            self[alias_key] = packet

        for block_name, block in self.graph.blocks.items():
            for output_name in block.output_names:
                output_key = GraphIO(block_name=block_name, block_port=output_name, io_type="output")
                self[output_key] = block.outputs[output_name]()

        for link in self.graph.links:
            output_key = GraphIO(block_name=link.src_name, block_port=link.src_output, io_type="output")
            input_key = GraphIO(block_name=link.dst_name, block_port=link.dst_input, io_type="input")
            self[input_key] = self[output_key]

        for block_name, block in self.graph.blocks.items():
            for input_name in block.input_names:
                input_key = GraphIO(block_name=block_name, block_port=input_name, io_type="input")
                if not input_key in self:
                    self[input_key] = block.inputs[input_name]()

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
            if data.trainability:
                self.register_parameter(f"{name}-{param_name}", data.value)

        return block

    def add_link(self, src_name: str, src_output: str,
                 dst_name: str, dst_input: str):
        if src_name not in self.blocks:
            raise KeyError(f"Add link failed - Block with name \"{src_name}\" does not exist")
        src_block = self.blocks[src_name]
        if src_output not in src_block.output_names:
            raise ValueError(f"Add link failed - Block \"{src_name}\" does not have output \"{src_output}\"")
        output_type = src_block.outputs[src_output]

        if dst_name not in self.blocks:
            raise KeyError(f"Add link failed - Block with name \"{dst_name}\" does not exist")
        dst_block = self.blocks[dst_name]
        if dst_input not in dst_block.input_names:
            raise ValueError(f"Add link failed - Block \"{dst_name}\" does not have input \"{dst_input}\"")
        input_type = dst_block.inputs[dst_input]

        if not output_type == input_type:
            raise TypeError("Add link failed - Input and output have different types")

        link = Link(src_name, src_output, dst_name, dst_input, output_type)
        self.links.append(link)
        self.nx.add_edge(src_name, dst_name, tailport=src_output, headport=dst_input)
        return link

    def write_params(self, name: str, **params):
        if name not in self.blocks:
            raise KeyError(f"Write params failed - Block \"{name}\" does not exist")

        block = self.blocks[name]

        for param, data in params.items():
            if param not in block.params:
                raise KeyError(f"Write param failed - Block \"{name}\" has no parameter \"{param}\"")

            block.params[param] = data
            if data.trainability:
                self.register_parameter(f"{name}-{param}", data.value)

    def write_requirements(self, name: str, **requirements):
        if name not in self.blocks:
            raise KeyError(f"Write requirements failed - Block \"{name}\" does not exist")

        block = self.blocks[name]

        block.requirements = requirements
        block.refresh()

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

        self.outputs: dict[str, type] = {}
        for alias, (src_name, src_output) in self.output_map.items():
            self.outputs[alias] = self.internal_graph.blocks[src_name].outputs[src_output]

            self.internal_graph.set_output(alias, src_name, src_output)

        self.inputs: dict[str, type] = {}
        for alias, (dst_name, dst_input) in self.input_map.items():
            self.inputs[alias] = self.internal_graph.blocks[dst_name].inputs[dst_input]
            self.internal_graph.set_input(alias, dst_name, dst_input)

        self.input_names = tuple(self.inputs.keys())
        self.output_names = tuple(self.outputs.keys())

    def refresh(self):
        self.internal_graph = self.generate_internal_graph()

    def generate_internal_graph(self) -> Graph:
        raise NotImplementedError

    def compute(self, inputs: dict[str, Packet]) -> dict[str, Packet]:
        return self.internal_graph.compute(**inputs)
