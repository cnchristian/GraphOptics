from primatives import Block, BlockParam, Graph, TRAINABLE, NOT_TRAINABLE
from utilities import draw_graph
from controls_blocks import GainBlock, SubtractBlock, SplitBlock, FeedbackSuperBlock

import torch


g = Graph()

g.add_block("feedback", FeedbackSuperBlock)

g.set_input("sys_in", "feedback", "i")
g.set_output("sys_out", "feedback", "o")

draw_graph(g, "graph.png")

# TODO this is not recognizing parameters of internal sub-graphs
optimizer = torch.optim.Adam(g.parameters(), lr=1e-2)
inp = torch.tensor(1, dtype=torch.complex64)
target = {"sys_out": torch.tensor(0.33, dtype=torch.complex64)}
loss_fn = torch.nn.L1Loss()

for epoch in range(100):
    optimizer.zero_grad()
    out = g.compute(sys_in=inp)
    loss = sum(loss_fn(out[k], target[k]) for k in out.keys())
    print(f"loss at epoch {epoch}: {loss}")
    loss.backward()
    optimizer.step()

print(f"final loss is {loss}")