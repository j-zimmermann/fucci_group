from FUCCIDataLoader import FUCCIDataset
import torch
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt

source_channels = (0, 1, 3)
target_channels = 0

validation_data = FUCCIDataset(
    root_dir="JuliusValidate",
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True,
)

# choose random frame
idx = np.random.randint(len(validation_data))
idx = 42
image, mask = validation_data[idx]
# unsqueeze because of expected shape: batch, channel, y, x
image = torch.unsqueeze(image, dim=0)


# load unets
preds = []
for i in range(1, 4):
    unet = UNet(
        depth=4,
        in_channels=i,
        out_channels=1,
        final_activation="Sigmoid",  # needed for masks
        num_fmaps=16,
    )
    # switch from training mode
    unet.eval()
    unet.load_state_dict(torch.load(f"semantic_{i}_channel_model/training_state.pt"))
    if i == 1:
        image_tmp = image[:, 1, ...]
        image_tmp = torch.unsqueeze(image_tmp, dim=1)
    elif i == 2:
        image_tmp = image[:, 0::2, ...]
    else:
        image_tmp = image
    print(image.shape)
    print(image_tmp.shape)
    preds.append(np.squeeze(unet(image_tmp).detach().numpy()) > 0.5)

image = np.squeeze(image)
mask = np.squeeze(mask.numpy())

fig, axs = plt.subplots(nrows=2, ncols=3, layout=None)
axs[0, 0].imshow(image[1], cmap="gray")
axs[0, 0].set_title("Tubulin")
axs[0, 1].imshow(image[0], cmap="gray")
axs[0, 1].set_title("Magenta")
axs[0, 2].imshow(image[2], cmap="gray")
axs[0, 2].set_title("Cyan")
for i in range(3):
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])


for i in range(1, 4):
    axs[1, i - 1].imshow(preds[i - 1])
    axs[1, i - 1].set_xticks([])
    axs[1, i - 1].set_yticks([])
    axs[1, i - 1].set_title(f"#CH: {i}")

plt.show()

fig, axs = plt.subplots(nrows=1, ncols=3, layout=None)
axs[0].imshow(image[1], cmap="gray")
axs[1].imshow(image[0], cmap="gray")
axs[2].imshow(image[2], cmap="gray")
plt.show()
