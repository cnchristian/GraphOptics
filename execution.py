from dataclasses import dataclass

from primatives import GraphState, GraphIO, Packet, is_empty, EMPTY_VALUE

import torch
import networkx as nx

def execute_block(graph, state, block_name: str) -> dict[str, Packet]:
    block = graph.blocks[block_name]
    input_dict = {}

    for io, value in state:
        if io.block_name == block_name and io.io_type == "input":
            input_dict[io.block_port] = state[io]

    output_dict = block.compute(input_dict)
    return output_dict

def execute_acyclic_region(graph, region):
    for block_name in region.block_names:
        output_dict = execute_block(graph, graph.state, block_name)

        for output in output_dict:
            output_key = GraphIO(block_name=block_name, block_port=output, io_type="output")
            graph.state[output_key] = output_dict[output]

def execute_cyclic_region(graph, region):
    packets, flat = graph.state.flatten()
    flat_out = CyclicRegionDEQ.apply(graph, region, packets, *flat)
    graph.state = GraphState.unflatten(graph, packets, flat_out, graph.state.index)


@dataclass
class ExecutionRegion:
    is_cyclic: bool
    block_names: list

def get_execution_regions(graph) -> list[ExecutionRegion]:
    scc_graph = nx.DiGraph()

    sccs = list(nx.strongly_connected_components(graph.nx))
    for i in range(len(sccs)):
        scc_graph.add_node(i)

    node_to_scc = {}
    for idx, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = idx

    for u, v in graph.nx.edges():
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        if scc_u != scc_v:                      # TODO does this break actually intentional cycles?
            scc_graph.add_edge(scc_u, scc_v)

    sorted_scc_ids = list(nx.topological_sort(scc_graph))

    def scc_is_cyclic(scc: set[str]) -> bool:
        node = next(iter(scc))
        return len(scc) != 1 or graph.nx.has_edge(node, node)

    regions = []
    for scc_id in sorted_scc_ids:
        scc_nodes = list(sccs[scc_id])
        is_cyclic = scc_is_cyclic(sccs[scc_id])
        regions.append(ExecutionRegion(is_cyclic=is_cyclic, block_names=sorted(scc_nodes)))

    return regions

def execute(graph, inputs) -> GraphState:
    graph.state.reset(inputs)

    execution_regions = get_execution_regions(graph)
    for execution_region in execution_regions:
        if execution_region.is_cyclic:
            execute_cyclic_region(graph, execution_region)
        else:
            execute_acyclic_region(graph, execution_region)

    return graph.state


def _update_state(graph, state, region):
    new_state = torch.clone(state)
    for block_name in region.block_names:
        output_dict = execute_block(graph, state, block_name)

        for output in output_dict:
            output_key = GraphIO(block_name=block_name, block_port=output, io_type="output")
            new_state[output_key] = output_dict[output]

    step = torch.sub(new_state, state)
    return step

def _broyden_solve(graph, region, tol=1e-5, max_iters=2000):
    def _inner(a: GraphState, b: GraphState) -> torch.Tensor:
        return sum([packet.value for packet in (torch.sum(torch.conj(a) * b)).packets])

    # TODO needs to stop double counting packets because of multiple indices mapping to same key
    # TODO needs to give maximum across batch instead of just averaging across batch
    def _avg_change(region: ExecutionRegion, Fz: GraphState, z: GraphState) -> torch.Tensor:
        total = 0
        for (k1, f_packet), (k2, z_packet) in zip(Fz, z):
            if not (k1.block_name in region.block_names and k2.block_name in region.block_names):
                continue

            f = f_packet.value
            base = z_packet.value
            if base.numel() == 0:
                base = torch.zeros_like(f) # TODO make sure this isn't making things complex when they should be floats
            ratio = f / base

            zero_zero_mask = (base == 0) & (f == 0)
            max_mean_ratio = torch.max(torch.abs(torch.mean(torch.where(zero_zero_mask, torch.zeros_like(ratio), ratio), dim=tuple(range(1, ratio.ndim)))))
            total += max_mean_ratio

        return torch.abs(total / len(z.packets))

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

    z = torch.clone(graph.state)
    Fz = _update_state(graph, z, region)

    for k in range(max_iters):
        z += Fz
        Fz = _update_state(graph, z, region)

        avg_change = _avg_change(region, Fz, z)
        print(f"{k}: {avg_change.detach().numpy()}")
        if avg_change < tol:
            return z

    print("Warning -- Broyden did not converge")
    return z
    """
    H = InverseJacobian(alpha=1.0, max_updates=50)

    for k in range(max_iters):
        res_norm = torch.sqrt(_inner(Fz, Fz).real)
        print(f"{k}: {res_norm.numpy()}")
        if res_norm < tol:
            return z

        p = torch.neg(H.apply(Fz))

        z_new = z + p
        Fz_new = _update_state(graph, z_new, region)

        s = p  # z_{k+1} - z_k
        y = Fz_new - Fz
        Hy = H.apply(y)
        denom = _inner(Hy, y)

        if torch.abs(denom) > 1e-12:
            H.add_update((s - Hy) / denom, Hy)

        z, Fz = z_new, Fz_new

    print("Warning -- Broyden did not converge")
    return z
    """

"""
orig_make_grads = torch.autograd._make_grads

def debug_make_grads(outputs, grads, *args, **kwargs):
    print("\n--- DEBUG _make_grads ---")
    print("is_grads_batched:", kwargs.get("is_grads_batched", None))

    for i, (o, g) in enumerate(zip(outputs, grads)):
        print(f"\nIndex {i}")
        print("  output shape:", tuple(o.shape))
        print("  grad shape:  ", None if g is None else tuple(g.shape))
        print("  output id:   ", id(o))
        print("  grad id:     ", None if g is None else id(g))

        if g is not None:
            print("  same object: ", o is g)
            print("  same storage:", o.data_ptr() == g.data_ptr())

        try:
            eq = (o == g) if g is not None else None
            print("  equality result shape:", None if eq is None else tuple(eq.shape))
        except Exception as e:
            print("  equality comparison error:", e)

    return orig_make_grads(outputs, grads, *args, **kwargs)

torch.autograd._make_grads = debug_make_grads
"""
class CyclicRegionDEQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, region, packets, *flat_state):
        z0 = GraphState.unflatten(graph, packets, flat_state, graph.state.index)
        with torch.no_grad():
            z_star = _broyden_solve(graph, region)
        packets, flat_out = z_star.flatten()

        for new_packet, old_packet in zip(packets, graph.state.packets):
            old_packet.data = new_packet.data

        ctx.graph = graph
        ctx.region = region
        ctx.packets = packets
        ctx.flat_state = flat_state

        return flat_out

    @staticmethod
    def backward(ctx, *grad_flat):
        graph = ctx.graph
        region = ctx.region
        packets = ctx.packets
        flat_state = ctx.flat_state

        z0 = GraphState.unflatten(graph, packets, flat_state, graph.state.index)
        with torch.enable_grad():
            z_star = _broyden_solve(graph, region)
            f_z = _update_state(graph, z_star, region)

        outputs = f_z.flatten()[1]
        inputs = z_star.flatten()[1]
        grad_flat = tuple([g if not is_empty(g) else torch.zeros_like(o) for g, o in zip(grad_flat, outputs)])
        grad_z_star = GraphState.unflatten(graph, packets, grad_flat, graph.state.index)
        for _, v in grad_z_star:
            v.value.requires_grad_(True)

        # TODO
        #  The first time that the empty elements of grad_outputs are set to 0, they take the correct dtype.
        #  After they have gone through grads, though, they end up with a complex dtype
        #  Manually setting back to a float destroys the gradients
        #  So there must be a way to fix the fact that they are becoming complex,
        #  Or they just need to be completely ignored because they are not part of the region
        #  Just removing them seems to make higher order gradients break
        """
        empty_mask = [is_empty(t) for t in grad_z_star.flatten()[1]]
        def Jt_v(v):
            print("calculating Jt")
            outputs = f_z.flatten()[1]
            inputs = z_star.flatten()[1]
            grad_outputs = v.flatten()[1]

            # Remove elements where empty_mask is True
            outputs_filtered = tuple(t for t, keep in zip(outputs, empty_mask) if not keep)
            inputs_filtered = tuple(t for t, keep in zip(inputs, empty_mask) if not keep)
            grad_outputs_filtered = tuple(t for t, keep in zip(grad_outputs, empty_mask) if not keep)

            grads_filtered = torch.autograd.grad(
                outputs=outputs_filtered,
                inputs=inputs_filtered,
                grad_outputs=grad_outputs_filtered,
                retain_graph=True,
                allow_unused=True,
                create_graph=True
            )

            filtered_iter = iter(grads_filtered)
            grads = tuple(next(filtered_iter) if not remove else EMPTY_VALUE for remove in empty_mask)
            out = GraphState.unflatten(graph, z_star.packets, list(grads), graph.state.index)
            return out
        """
        def Jt_v(v):
            grad_outputs = v.flatten()[1]
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
                allow_unused=True,
                create_graph=True
            )

            out = GraphState.unflatten(graph, z_star.packets, list(grads), graph.state.index)
            return out


        # Fixed-point solve for implicit gradient
        v = grad_z_star
        for _ in range(50):             # TODO this should probably be dynamic to convergence, not hardcoded -- also value just explodes?
            v = grad_z_star + Jt_v(v)

        # TODO - since the graph object here is not actually the main graph object, the gradients of the params do not end up getting set correctly
        #  why did this work previously?
        # Parameter gradients
        params = []
        for block_name in region.block_names:
            block = graph.blocks[block_name]
            if hasattr(block, "params"):
                params.extend(list([param._value for param in block.params.values() if param.trainable]))

        if params:
            param_grads = torch.autograd.grad(
                outputs=f_z.flatten()[1],
                inputs=params,
                grad_outputs=v.flatten()[1],
                retain_graph=False
            )

            for p, g in zip(params, param_grads):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad = p.grad + g

        # TODO
        #  this does successfully set the gradients for this copy of the graph
        return (None, None, None, *(v.flatten()[1]))
