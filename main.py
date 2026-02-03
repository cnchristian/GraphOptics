from primatives import BlockParam, Graph, TRAINABLE, NOT_TRAINABLE
from utilities import draw_graph
from optics_blocks import PropagationBlock, MirrorBlock, SLMBlock

import matplotlib.pyplot as plt

import torch
from torch.nn import Parameter
import numpy as np

prop_dist = BlockParam(Parameter(torch.tensor([5e-3], dtype=torch.float)), TRAINABLE)

g = Graph()

g.add_block("mirror", MirrorBlock)

g.add_block("forward_propagation", PropagationBlock)
g.write_param("forward_propagation", "distance", prop_dist)

g.add_block("slm", SLMBlock)

g.add_block("backward_propagation", PropagationBlock)
g.write_param("backward_propagation", "distance", prop_dist)

g.add_link("mirror", "o2", "forward_propagation", "i")
g.add_link("forward_propagation", "o", "slm", "i")
g.add_link("slm", "o", "backward_propagation", "i")
g.add_link("backward_propagation", "o", "mirror", "i2")

g.set_input("input_field", "mirror", "i1")
g.set_input("slm_phase", "slm", "phase")
g.set_output("output_field", "mirror", "o1")

input_field = torch.ones((600, 600), dtype=torch.complex64)

phase = torch.zeros((600, 600), dtype=torch.complex64)
phase[250:300, 250:300] = torch.pi
phase[300:350, 300:350] = torch.pi

out = g.compute(input_field=input_field, slm_phase=phase)

plt.imshow(np.abs(input_field.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(phase.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(out["output_field"].detach().numpy()))
plt.colorbar()
plt.show()

draw_graph(g, "graph.png")
"""
target_field = torch.from_numpy(np.load("output_field.npy"))
target = {"output_field": target_field}

optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)
loss_fn = torch.nn.L1Loss()

for epoch in range(30):
    optimizer.zero_grad()
    out = g.compute(input_field=input_field)
    loss = sum(loss_fn(out[k], target[k]) for k in out.keys())
    print(f"loss at epoch {epoch}: {loss}")
    loss.backward()
    optimizer.step()

print(f"final loss is {loss}")
plt.imshow(np.abs(target_field.detach().numpy()))
plt.colorbar()
plt.show()
"""
