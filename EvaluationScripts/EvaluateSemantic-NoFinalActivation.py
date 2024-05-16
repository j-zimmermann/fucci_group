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

source_channels = (0, 3)
target_channels = 0


unet = UNet(depth=4,
            in_channels=len(source_channels),
            out_channels=1,
            final_activation=None,  # needed for masks
            num_fmaps=16)


# switch from training mode
unet.eval()

unet.load_state_dict(torch.load("semantic_2_channel_model/training_state.pt"))

# TODO make it useful
validation_data = FUCCIDataset(
    root_dir="Julius",
    transform=transforms.RandomCrop(256),
    # 0 = mean and 1 = variance
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True
)

idx = np.random.randint(len(validation_data))
image, mask = validation_data[idx]

# unsqueeze because of expected shape: batch, channel, y, x
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image)
mask = np.squeeze(mask.numpy())
pred_plot = np.squeeze(pred.detach().numpy())


plot_four(image[0], image[1], mask, pred_plot)
plt.savefig("Presentation.png")

pred_plot.max()


