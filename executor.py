from dataclasses import dataclass
import networkx as nx

import torch
from primatives import GraphState, GraphIO

class GraphExecutor:
    def __init__(self, graph):
        self.graph = graph
        self.state = GraphState(graph)

    def execute(self, inputs) -> GraphState:
        self.state.reset(inputs)

        execution_regions = get_execution_regions(self.graph)
        for execution_region in execution_regions:
            if execution_region.type == "acyclic":
                self.execute_acyclic_region(execution_region)
            elif execution_region.type == "cyclic":
                self.execute_cyclic_region(execution_region)
            else:
                raise ValueError(f"Execution failed - unsupported execution region type \"{execution_region}\"")

        return self.state

    def execute_block(self, block_name: str, state=None):
        state = state if state is not None else self.state
        block = self.graph.blocks[block_name]
        input_dict = {}

        for io, value in state:
            if io.block_name == block_name and io.io_type == "input":
                input_dict[io.block_port] = state[io]

        output_dict = block.compute(input_dict)
        return output_dict

    def execute_acyclic_region(self, region):
        for block_name in region.block_names:
            output_dict = self.execute_block(block_name)

            for output in output_dict:
                output_key = GraphIO(block_name=block_name, block_port=output, io_type="output")
                self.state[output_key] = output_dict[output]

    def execute_cyclic_region(self, region):
        class InverseJacobian:
            def __init__(self, alpha: float = 1.0, max_updates: int = None):
                self.alpha = alpha
                self.updates: list[tuple[GraphState, GraphState]] = []
                self.max_updates = max_updates

            def apply(self, x: GraphState) -> GraphState:
                y = self.alpha * x
                for u, v in self.updates:
                    coeff = _inner(v, x)
                    y = torch.add(y, coeff * u)
                return y

            def add_update(self, u: list[torch.Tensor], v: list[torch.Tensor]):
                self.updates.append((torch.clone(u), torch.clone(v)))
                if self.max_updates is not None:
                    self.updates = self.updates[-self.max_updates:]

        def _inner(a: GraphState, b: GraphState) -> torch.Tensor:
            return sum((torch.sum(torch.conj(a) * b)).values)

        def _update_state(state):
            new_state = torch.clone(state)
            for block_name in region.block_names:
                output_dict = self.execute_block(block_name, state)

                for output in output_dict:
                    output_key = GraphIO(block_name=block_name, block_port=output, io_type="output")
                    new_state[output_key] = output_dict[output]

            step = torch.sub(new_state, state)
            return step

        def _broyden_solve(x0: GraphState, tol=1e-5, max_iters=250):
            x = torch.clone(x0)
            Fx = _update_state(x)
            H = InverseJacobian(alpha=1.0, max_updates=50)

            for k in range(max_iters):
                res_norm = torch.sqrt(_inner(Fx, Fx).real)
                if res_norm < tol:
                    return x

                p = torch.neg(H.apply(Fx))

                x_new = x + p
                Fx_new = _update_state(x_new)

                s = p  # z_{k+1} - z_k
                y = Fx_new - Fx
                Hy = H.apply(y)
                denom = _inner(Hy, y)

                if torch.abs(denom) < 1e-12:
                    x, Fx = x_new, Fx_new
                    continue

                H.add_update((s - Hy) / denom, Hy)
                x, Fx = x_new, Fx_new

        with torch.no_grad():
            z_state = _broyden_solve(self.state)
        self.state = z_state


@dataclass
class ExecutionRegion:
    type: str                 # "acyclic", "cyclic"
    block_names: list         # list of block names (strings)

def get_execution_regions(graph) -> list[ExecutionRegion]:
    sccs = list(nx.strongly_connected_components(graph.nx))

    # Map each node to its SCC ID
    node_to_scc = {}
    for idx, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = idx

    # -----------------------------
    # 2. Build SCC graph (DAG)
    # -----------------------------
    scc_graph = nx.DiGraph()
    for i in range(len(sccs)):
        scc_graph.add_node(i)

    for u, v in graph.nx.edges():
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        if scc_u != scc_v:
            scc_graph.add_edge(scc_u, scc_v)

    # -----------------------------
    # 3. Topologically sort SCC graph
    # -----------------------------
    sorted_scc_ids = list(nx.topological_sort(scc_graph))

    # -----------------------------
    # 4. Tag SCC types
    # -----------------------------
    def classify_scc(scc: set[str]) -> str:
        node = next(iter(scc))
        if len(scc) == 1 and not graph.nx.has_edge(node, node):
            return "acyclic"
        else:
            return "cyclic"

    # -----------------------------
    # 5. Build execution region objects
    # -----------------------------
    regions = []
    for scc_id in sorted_scc_ids:
        scc_nodes = list(sccs[scc_id])
        rtype = classify_scc(sccs[scc_id])
        regions.append(ExecutionRegion(type=rtype, block_names=sorted(scc_nodes)))

    return regions