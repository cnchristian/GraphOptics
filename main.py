from primatives import BlockParam, Graph, TRAINABLE, NOT_TRAINABLE
from utilities import draw_graph
from optics_toolkit import PropagationBlock, MirrorBlock, SLMBlock, FieldPacket
from image_toolkit import PadBlock, ImagePacket

import matplotlib.pyplot as plt

import torch
from torch.nn import Parameter
import numpy as np

prop_dist = BlockParam(Parameter(torch.tensor([5e-3], dtype=torch.float)), TRAINABLE)
slm_width = BlockParam(Parameter(torch.tensor(1920, dtype=torch.int)), NOT_TRAINABLE)
slm_height = BlockParam(Parameter(torch.tensor(1152, dtype=torch.int)), NOT_TRAINABLE)

g = Graph()

# FP Cavity
g.add_block("mirror", MirrorBlock)

g.add_block("forward_propagation", PropagationBlock)
g.write_params("forward_propagation", distance=prop_dist)

g.add_block("slm", SLMBlock)
g.write_requirements("slm", width=slm_width.value, height=slm_height.value)

g.add_block("backward_propagation", PropagationBlock)
g.write_params("backward_propagation", distance=prop_dist)

g.add_link("mirror", "o2", "forward_propagation", "i")
g.add_link("forward_propagation", "o", "slm", "i")
g.add_link("slm", "o", "backward_propagation", "i")
g.add_link("backward_propagation", "o", "mirror", "i2")

# Phase Encoding
g.add_block("input_to_slm", PadBlock)
g.write_params("input_to_slm", padded_width=slm_width, padded_height=slm_height)

g.add_link("input_to_slm", "o", "slm", "phase")

# IO Configuration
g.set_input("input_field", "mirror", "i1")
g.set_input("phase_img", "input_to_slm", "phase")
g.set_output("output_field", "mirror", "o1")


input_field_data = {
    "height": int(slm_height.value.data),
    "width": int(slm_width.value.data),
    "ds": 9.2e-6,
    "wavelength": 633e-9,
}
input_field = torch.ones(input_field_data["height"], input_field_data["width"], dtype=torch.complex64)
input_packet = FieldPacket(data=input_field_data, value=input_field)

phase_image_data = {
    "height": 600,
    "width": 600,
    "min_value": 0,
    "max_value": 2*np.pi,
}
phase_image = torch.zeros(phase_image_data["height"], phase_image_data["width"], dtype=torch.complex64)
phase_image[250:300, 250:300] = torch.pi
phase_image[300:350, 300:350] = torch.pi
phase_packet = ImagePacket(data=phase_image_data, value=phase_image)

out = g.compute(input_field=input_packet, slm_phase=phase_packet)

plt.imshow(np.abs(input_field.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(phase_image.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(out["output_field"].detach().numpy()))
plt.colorbar()
plt.show()

draw_graph(g, "graph.png")
