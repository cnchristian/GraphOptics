from primatives import RealParam, Graph, IntParam
from utilities import draw_graph
from optics_toolkit import PropagationBlock, MirrorBlock, SLMBlock, FieldPacket, LensBlock, CameraBlock
from image_toolkit import PadBlock, ImagePacket

import torch
import numpy as np
import matplotlib.pyplot as plt

upscaling = 4

mirror_reflectance = RealParam(0.5)
prop_dist = RealParam(2.5e-3)
slm_width = IntParam(1920*upscaling)
slm_height = IntParam(1152*upscaling)
slm_resolution = RealParam(9.2e-6/upscaling)
slm_undiffracted = RealParam(0.05)
beam_splitter_ratio = RealParam(0.5)
l1_focal_length = RealParam(100e-3)
l2_focal_length = RealParam(35e-3)

camera_width = IntParam(1440)
camera_height = IntParam(1080)
camera_ds = RealParam(3.45e-6)

g = Graph()

# FP Cavity
g.add_block("mirror", MirrorBlock)
g.write_params("mirror", reflectance=mirror_reflectance)

g.add_block("forward_propagation", PropagationBlock)
g.write_params("forward_propagation", distance=prop_dist)

g.add_block("slm", SLMBlock)
g.write_requirements("slm", width=slm_width.value, height=slm_height.value)
g.write_params("slm", undiffracted=slm_undiffracted)

g.add_block("backward_propagation", PropagationBlock)
g.write_params("backward_propagation", distance=prop_dist)

g.add_link("mirror", "o2", "forward_propagation", "i")
g.add_link("forward_propagation", "o", "slm", "i")
g.add_link("slm", "o", "backward_propagation", "i")
g.add_link("backward_propagation", "o", "mirror", "i2")

# Imaging Arm
g.add_block("L1", LensBlock)
g.add_block("L1_prop_1_forward", PropagationBlock)
g.add_block("L1_prop_2_forward", PropagationBlock)
g.write_params("L1_prop_1_forward", distance=l1_focal_length)
g.write_params("L1", focal_length=l1_focal_length)
g.write_params("L1_prop_2_forward", distance=l1_focal_length)

g.add_block("L2", LensBlock)
g.add_block("L2_prop_1_forward", PropagationBlock)
g.add_block("L2_prop_2_forward", PropagationBlock)
g.write_params("L2_prop_1_forward", distance=l2_focal_length)
g.write_params("L2", focal_length=l2_focal_length)
g.write_params("L2_prop_2_forward", distance=l2_focal_length)

g.add_link("mirror", "o1", "L1_prop_1_forward", "i")
g.add_link("L1_prop_1_forward", "o", "L1", "i")
g.add_link("L1", "o", "L1_prop_2_forward", "i")
g.add_link("L1_prop_2_forward", "o", "L2_prop_1_forward", "i")
g.add_link("L2_prop_1_forward", "o", "L2", "i")
g.add_link("L2", "o", "L2_prop_2_forward", "i")

g.add_block("camera", CameraBlock)
g.write_params("camera", camera_width=camera_width, camera_height=camera_height, camera_ds=camera_ds)
g.add_link("L2_prop_2_forward", "o", "camera", "i")

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
    "ds": slm_resolution,
    "wavelength": RealParam(561e-9),
}
input_field = torch.ones((slm_height.val(), slm_width.val()), dtype=torch.complex64)
input_packet = FieldPacket(data=input_field_data, value=input_field)

phase_image_data = {
    "height": slm_height,
    "width": slm_width,
    "min_value": RealParam(0),
    "max_value": RealParam(2 * np.pi),
}

f = 0.05                  # 5 cm focal length
dx = input_field_data["ds"].val()
k = 2*np.pi/input_field_data["wavelength"].val()
H = phase_image_data["height"].val()
W = phase_image_data["width"].val()
y = (torch.arange(H) - H/2) * dx
x = (torch.arange(W) - W/2) * dx
Y, X = torch.meshgrid(y, x, indexing='ij')
#phase_image = torch.remainder(-(k / (2 * f)) * (X**2 + Y**2), 2 * np.pi)

slm_curvature = RealParam(-(k / (2 * 25)) * (X**2 + Y**2))
g.write_params("slm", biases=slm_curvature)

phase_image = torch.zeros((phase_image_data["height"].val(), phase_image_data["width"].val()), dtype=torch.complex64)
phase_image[(576-150)*upscaling:(576)*upscaling, (960)*upscaling:(960+150)*upscaling] = torch.pi
phase_image[(576)*upscaling:(576+150)*upscaling, (960-150)*upscaling:(960)*upscaling] = torch.pi

phase_packet = ImagePacket(data=phase_image_data, value=phase_image)

draw_graph(g, "graph.png")
out = g.compute(input_field=input_packet, phase_img=phase_packet)


plt.imshow(np.abs(out["output_field"].value.detach().numpy()))
plt.colorbar()
plt.show()
