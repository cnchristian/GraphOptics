from primatives import RealParam, Graph, IntParam
from utilities import draw_graph
from optics_toolkit import PropagationBlock, MirrorBlock, SLMBlock, FieldPacket
from image_toolkit import PadBlock, ImagePacket

import torch
import numpy as np
import matplotlib.pyplot as plt


prop_dist = RealParam(5e-3)
slm_width = IntParam(1920)
slm_height = IntParam(1152)

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
g.set_input("phase_img", "input_to_slm", "i")
g.set_output("output_field", "mirror", "o1")


input_field_data = {
    "height": slm_height,
    "width": slm_width,
    "ds": RealParam(9.2e-6),
    "wavelength": RealParam(633e-9),
}
input_field = torch.ones((slm_height.val(), slm_width.val()), dtype=torch.complex64)
input_packet = FieldPacket(data=input_field_data, value=input_field)

phase_image_data = {
    "height": IntParam(600),
    "width": IntParam(600),
    "min_value": RealParam(0),
    "max_value": RealParam(2 * np.pi),
}
phase_image = torch.zeros((phase_image_data["height"].val(), phase_image_data["width"].val()), dtype=torch.complex64)
phase_image[150:300, 150:300] = torch.pi
phase_image[300:450, 300:450] = torch.pi
phase_packet = ImagePacket(data=phase_image_data, value=phase_image)

out = g.compute(input_field=input_packet, phase_img=phase_packet)

plt.imshow(np.abs(input_field.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(phase_image.detach().numpy()))
plt.colorbar()
plt.show()
plt.imshow(np.abs(out["output_field"].value.detach().numpy()))
plt.colorbar()
plt.show()

draw_graph(g, "graph.png")
