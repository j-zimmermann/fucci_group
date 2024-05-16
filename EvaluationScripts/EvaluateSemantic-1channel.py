from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, plot_three
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

source_channels = (1, )
target_channels = 0


unet = UNet(depth=4,
            in_channels=len(source_channels),
            out_channels=1,
            final_activation="Sigmoid",  # needed for masks
            num_fmaps=16)


# switch from training mode
unet.eval()

unet.load_state_dict(torch.load("semantic_1_channel_model/training_state.pt"))

# TODO make it useful
validation_data = FUCCIDataset(
    root_dir="JuliusValidate",
    transform=transforms.RandomCrop(1024),
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
plot_three(image, mask, pred_plot > 0.5)
plt.show()
plt.imshow(np.log(pred_plot))
plt.show()
plt.imshow(pred_plot > 0.5)
plt.show()
