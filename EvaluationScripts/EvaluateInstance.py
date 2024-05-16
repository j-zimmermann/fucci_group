from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, plot_four, get_labels_from_prediction
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

source_channels = (0, 3)
target_channels = 0


unet = UNet(depth=4,
            in_channels=len(source_channels),
            out_channels=2,
            final_activation="Sigmoid",  # needed for masks
            num_fmaps=16)


# switch from training mode
unet.eval()

unet.load_state_dict(torch.load("instance_2_channel_model/training_state.pt"))

validation_data = FUCCIDataset(
    root_dir="Julius",
    #transform=transforms.RandomCrop(2028),
    source_transform=None,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    normalize=True,
    semantic=False
)

validation_loader = DataLoader(validation_data, batch_size=2,
        shuffle=True, num_workers=8)


idx = np.random.randint(len(validation_data))
image, mask = validation_data[idx]

id_offset = 0
min_seed_distance = 3 
# unsqueeze because of expected shape: batch, channel, y, x
pred = unet(torch.unsqueeze(image, dim=0))
labels = get_labels_from_prediction(pred, id_offset, min_seed_distance)
gt_labels = get_labels_from_prediction(mask, id_offset, min_seed_distance)

image = np.squeeze(image)
mask = np.squeeze(mask.numpy())
pred_plot = np.squeeze(pred.detach().numpy())
plot_four(image[0], image[1], mask[0] + mask[1], pred_plot[0] + pred_plot[1])
plt.show()

plot_four(image[0], image[1], gt_labels, labels)
plt.show()

c = plt.imshow(pred_plot[0])
plt.colorbar(c)
plt.show()

plt.imshow(pred_plot[0] + pred_plot[1])
c = plt.imshow(mask[1])
plt.show()
