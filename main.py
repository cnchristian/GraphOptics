from torch.utils.data import DataLoader, Subset

from primatives import RealParam, Graph, IntParam, UnitParam, PositiveParam
from utilities import draw_graph
from optics_toolkit import PropagationBlock, MirrorBlock, SLMBlock, FieldPacket, CameraBlock, FourFBlock
from image_toolkit import PadBlock, ImagePacket, TessellateBlock, ValueScaleBlock, ScaleBlock, FragmentBlock, CropBlock

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

upscaling = 4

mirror_reflectance = RealParam(0)
prop_dist = PositiveParam(2.5e-3)
slm_width = IntParam(400*upscaling)
slm_height = IntParam(400*upscaling)
slm_resolution = RealParam(9.2e-6/upscaling)
slm_undiffracted = UnitParam(0.05)
beam_splitter_ratio = UnitParam(0.5)
l1_focal_length = RealParam(100e-3)
l2_focal_length = RealParam(35e-3)
mask = RealParam(torch.ones(slm_height.val(), slm_width.val())) # TODO this doesn't quite work as a unitparam because it is 1 inclusively
wavelength = PositiveParam(561e-9)

mask_resolution = 561e-9 * 100e-3 / 9.2e-6

camera_width = IntParam(720//2)
camera_height = IntParam(540//2)
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
slm_curvature.set_trainable(True)
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


# TODO maybe these need to require grad for this to work?
input_field_data = {
    "height": slm_height,
    "width": slm_width,
    "ds": slm_resolution,
    "wavelength": wavelength,
}
input_field = torch.ones((slm_height.val(), slm_width.val()), dtype=torch.complex64).unsqueeze(0).repeat(2, 1, 1)
input_field.requires_grad_(True)
input_packet = FieldPacket(data=input_field_data, value=input_field)

img_data = {
    "height": IntParam(28),
    "width": IntParam(28),
    "min_value": RealParam(0),
    "max_value": RealParam(255),
}

transform = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
"""
for img, label in mnist_train:
    if label == 0:
        img1 = np.array(img)  # Convert PIL image to NumPy
        break
for img, label in mnist_train:
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
"""

class compound_model(nn.Module):
    def __init__(self, graph, input_field):
        super().__init__()

        self.model = graph
        self.input_field = input_field

        in_features = 27**2
        self.LC = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=in_features, out_features=10, dtype=torch.float32),
        )

    def forward(self, x):
        x1 = self.model.compute(input_field=self.input_field, img=x)
        x2 = self.LC(x1["output_field"].value.unsqueeze(1))
        return x2



train_loader = DataLoader(Subset(mnist_train, range(0, 6)), batch_size=2, shuffle=True)
test_loader = DataLoader(Subset(mnist_test, range(0, 2)), batch_size=2, shuffle=True)

model = compound_model(g, input_packet)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
loss_function = torch.nn.CrossEntropyLoss()

initial_weights = g.blocks["slm"].params["weights"]._value.clone().detach().cpu().numpy()
initial_biases = g.blocks["slm"].params["biases"]._value.clone().detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Weights image ---
im0 = axes[0].imshow(initial_weights, cmap="viridis", aspect="auto")
axes[0].set_title("Initial Weights")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# --- Biases image ---
im1 = axes[1].imshow(initial_biases, cmap="magma", aspect="auto")
axes[1].set_title("Initial Biases")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

torch.autograd.set_detect_anomaly(True)
for epoch in range(5):
    model.train(True)
    running_train_loss = 0
    for step, (train_images, train_labels) in enumerate(train_loader):
        # Move the data to the GPU
        train_images, train_labels = train_images, train_labels

        train_packets = ImagePacket(data=img_data, value=train_images.squeeze(1))

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        train_outputs = model.forward(train_packets)

        # Calculate the loss
        train_loss = loss_function(train_outputs, train_labels)

        # Backward pass and optimize
        train_loss.backward()

        optimizer.step()  # Update model weights
        optimizer.zero_grad()  # Reset gradients

        running_train_loss += train_loss

    avg_train_loss = running_train_loss / len(train_loader)

    model.eval()
    running_test_loss = 0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images, test_labels

            test_packets = ImagePacket(data=img_data, value=test_images.squeeze(1))

            test_outputs = model.forward(test_packets)
            test_loss = loss_function(test_outputs, test_labels)
            running_test_loss += test_loss

    avg_test_loss = running_test_loss / len(test_loader)

    print(avg_train_loss, avg_test_loss)

final_weights = g.blocks["slm"].params["weights"]._value.clone().detach().cpu().numpy()
final_biases = g.blocks["slm"].params["biases"]._value.clone().detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Weights image ---
im0 = axes[0].imshow(final_weights, cmap="viridis", aspect="auto")
axes[0].set_title("Final Weights")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# --- Biases image ---
im1 = axes[1].imshow(final_biases, cmap="magma", aspect="auto")
axes[1].set_title("Final Biases")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
