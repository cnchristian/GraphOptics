from primatives import RealParam, Graph, IntParam, UnitParam, PositiveParam
from utilities import draw_graph
from optics_toolkit import PropagationBlock, MirrorBlock, SLMBlock, FieldPacket, CameraBlock, FourFBlock
from image_toolkit import PadBlock, ImagePacket, TessellateBlock, ValueScaleBlock, ScaleBlock, FragmentBlock, CropBlock

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets

upscaling = 4

mirror_reflectance = RealParam(0)
prop_dist = PositiveParam(2.5e-3)
slm_width = IntParam(1920*upscaling)
slm_height = IntParam(1152*upscaling)
slm_resolution = RealParam(9.2e-6/upscaling)
slm_undiffracted = UnitParam(0.05)
beam_splitter_ratio = UnitParam(0.5)
l1_focal_length = RealParam(100e-3)
l2_focal_length = RealParam(35e-3)
mask = RealParam(torch.ones(slm_height.val(), slm_width.val())) # TODO this doesn't quite work as a unitparam because it is 1 inclusively
wavelength = PositiveParam(561e-9)

camera_width = IntParam(720)
camera_height = IntParam(540)
camera_ds = PositiveParam(3.45e-6)

g = Graph()

# FP Cavity
g.add_block("mirror", MirrorBlock)
g.write_params("mirror", reflectance=mirror_reflectance)

g.add_block("forward_propagation", PropagationBlock)
g.write_params("forward_propagation", distance=prop_dist)

g.add_block("slm", SLMBlock)
g.write_requirements("slm", width=slm_width.val(), height=slm_height.val())
g.write_params("slm", undiffracted=slm_undiffracted)

g.add_block("backward_propagation", PropagationBlock)
g.write_params("backward_propagation", distance=prop_dist)

g.add_link("mirror", "o2", "forward_propagation", "i")
g.add_link("forward_propagation", "o", "slm", "i")
g.add_link("slm", "o", "backward_propagation", "i")
g.add_link("backward_propagation", "o", "mirror", "i2")

dx = slm_resolution.val()
k = 2*np.pi/wavelength.val()
H = slm_height.val()
W = slm_width.val()
y = (torch.arange(H) - H/2) * dx
x = (torch.arange(W) - W/2) * dx
Y, X = torch.meshgrid(y, x, indexing='ij')
slm_curvature = RealParam(-(k / (2 * 25)) * (X**2 + Y**2))
g.write_params("slm", biases=slm_curvature)

# Imaging Arm
g.add_block("4F", FourFBlock)
g.write_requirements("4F", width=slm_width.val(), height=slm_height.val())
g.write_params("4F", f1=l1_focal_length, f2=l2_focal_length, mask=mask)
g.add_link("mirror", "o1", "4F", "i")

g.add_block("camera", CameraBlock)
g.write_params("camera", camera_width=camera_width, camera_height=camera_height, camera_ds=camera_ds)
g.add_link("4F", "o", "camera", "i")

small_size_int = IntParam(27)
small_size_float = RealParam(28*35/100*9.2/3.45)

g.add_block("fragment", FragmentBlock)
g.write_params("fragment", individual_width=small_size_float, individual_height=small_size_float)
g.add_link("camera", "o", "fragment", "i")

# Phase Encoding
g.add_block("rescale", ValueScaleBlock)
g.write_params("rescale", min_value=RealParam(0), max_value=RealParam(2*np.pi))

g.add_block("upsample", ScaleBlock)
g.write_params("upsample", scale_factor=RealParam(upscaling))
g.add_link("rescale", "o", "upsample", "i")

g.add_block("input_to_slm", TessellateBlock)
g.write_params("input_to_slm", total_width=slm_width, total_height=slm_height)
g.add_link("upsample", "o", "input_to_slm", "i")

g.add_link("input_to_slm", "o", "slm", "phase")

# IO Configuration
g.set_input("input_field", "mirror", "i1")
g.set_input("img", "rescale", "i")
g.set_output("output_field", "fragment", "o")

input_field_data = {
    "height": slm_height,
    "width": slm_width,
    "ds": slm_resolution,
    "wavelength": wavelength,
}
input_field = torch.ones((slm_height.val(), slm_width.val()), dtype=torch.complex64).unsqueeze(0).repeat(2, 1, 1)
input_packet = FieldPacket(data=input_field_data, value=input_field)

img_data = {
    "height": IntParam(28),
    "width": IntParam(28),
    "min_value": RealParam(0),
    "max_value": RealParam(255),
}

mnist = datasets.FashionMNIST(root="./data", train=True, download=True)
for img, label in mnist:
    if label == 0:
        img1 = np.array(img)  # Convert PIL image to NumPy
        break
for img, label in mnist:
    if label == 1:
        img2 = np.array(img)  # Convert PIL image to NumPy
        break

img1 = torch.from_numpy(img1).type(torch.float32)
img2 = torch.from_numpy(img2).type(torch.float32)
img_packet = ImagePacket(data=img_data, value=torch.stack([img1, img2]))

draw_graph(g, "graph.png")
out = g.compute(input_field=input_packet, img=img_packet)

output_field = np.abs(out["output_field"].value.detach().numpy())
plt.imshow(output_field[0])
plt.colorbar()
plt.show()
plt.imshow(output_field[1])
plt.colorbar()
plt.show()