from primatives import BlockParam, Graph, TRAINABLE, NOT_TRAINABLE
from utilities import draw_graph
from optics_blocks import PropagationBlock, MirrorBlock

import matplotlib.pyplot as plt

import torch
from torch.nn import Parameter
import numpy as np

prop_dist = BlockParam(Parameter(torch.tensor([1e-4], dtype=torch.float)), TRAINABLE)

g = Graph()

g.add_block("mirror", MirrorBlock)

g.add_block("forward_propagation", PropagationBlock)
g.write_param("forward_propagation", "distance", prop_dist)

g.add_block("slm", MirrorBlock)
g.write_param("slm", "reflectance", BlockParam(Parameter(torch.tensor([1.0])), NOT_TRAINABLE))

g.add_block("backward_propagation", PropagationBlock)
g.write_param("backward_propagation", "distance", prop_dist)

g.add_link("mirror", "o2", "forward_propagation", "i")
g.add_link("forward_propagation", "o", "slm", "i2")
g.add_link("slm", "o2", "backward_propagation", "i")
g.add_link("backward_propagation", "o", "mirror", "i2")

g.set_input("input_field", "mirror", "i1")
g.set_output("output_field", "mirror", "o1")

draw_graph(g, "graph.png")

#input_field = torch.zeros((200, 300), dtype=torch.complex64)
#input_field[90:110, 140:160] = 1
input_field = torch.zeros((11, 11), dtype=torch.complex64)
input_field[5, 5] = 1

output_field = g.compute(input_field=input_field)["output_field"]
plt.imshow(np.abs(input_field.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(output_field.detach().numpy()))
plt.colorbar()
plt.show()

"""
output_field = torch.from_numpy(np.load("goal.npy"))

target = {"output_field": output_field}

optimizer = torch.optim.Adam(g.parameters(), lr=1e-5)
loss_fn = torch.nn.L1Loss()

for epoch in range(100):
    optimizer.zero_grad()
    out = g.compute(input_field=input_field)
    loss = sum(loss_fn(out[k], target[k]) for k in out.keys())
    print(f"loss at epoch {epoch}: {loss}")
    loss.backward()
    optimizer.step()

print(f"final loss is {loss}")

plt.imshow(np.abs(out["output_field"].detach().numpy()))
plt.show()
"""