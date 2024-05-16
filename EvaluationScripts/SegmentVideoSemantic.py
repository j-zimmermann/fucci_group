from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, plot_four
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.measure import label
import zarr

source_channels = (0, 3)
target_channels = 0
model_dir = "semantic_2_channel_model/training_state.pt"

unet = UNet(
    depth=4,
    in_channels=len(source_channels),
    out_channels=1,
    final_activation="Sigmoid",
    num_fmaps=16,
)


# switch from training mode
unet.eval()

unet.load_state_dict(torch.load(model_dir))

validation_data = FUCCIDataset(
    root_dir="JuliusValidate",
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True,
)

# use 8-bit unsigned int
labels = zarr.zeros(
    shape=(
        validation_data.open_videos[0].dims.T,
        validation_data.open_videos[0].dims.Y,
        validation_data.open_videos[0].dims.X,
    ),
    chunks=(validation_data.open_videos[0].dims.T,),
    dtype="uint8",
)
for idx in range(len(validation_data)):
    image, _ = validation_data[idx]
    # unsqueeze because of expected shape: batch, channel, y, x
    pred = unet(torch.unsqueeze(image, dim=0))
    labels[idx] = label(pred.squeeze() > 0.5)

zarr.save("test_labels.zarr", labels)
