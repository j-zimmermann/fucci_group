from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, plot_four, WeightedMSELoss
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

n_epochs = 16
model_directory = "semantic_2_channel_model"
if not os.path.isdir(model_directory):
    os.mkdir(model_directory)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()

source_channels = (0, 3)
target_channels = 0

print("Loading data")
# TODO add transforms
train_data = FUCCIDataset(
    root_dir="Julius",
    transform=transforms.RandomCrop(256),
    # 0 = mean and 1 = variance
    source_transform=None,
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True,
)

train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)

unet = UNet(
    depth=4,
    in_channels=len(source_channels),
    out_channels=1,
    final_activation="Sigmoid",  # needed for masks
    num_fmaps=16,
)
unet.to(device)


# loss = torch.nn.MSELoss()
loss = torch.nn.BCELoss()
# loss = WeightedMSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
# optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

print("Start training")
for epoch in tqdm(range(n_epochs)):
    train(unet, train_loader, optimizer, loss, epoch, device=device)
    if epoch % 5 == 0:
        torch.save(
            unet.state_dict(), os.path.join(model_directory, "training_state.pt")
        )

print("Showing example")
# switch from training mode
unet.eval()

# TODO make it useful
validation_data = FUCCIDataset(
    root_dir="Julius",
    transform=transforms.RandomCrop(256),
    # 0 = mean and 1 = variance
    source_transform=transforms.Normalize([0.0], [1.0]),
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
)

idx = np.random.randint(len(validation_data))
image, mask = validation_data[idx]

unet.to("cpu")
# unsqueeze because of expected shape: batch, channel, y, x
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image)
mask = np.squeeze(mask.numpy())
pred_plot = np.squeeze(pred.detach().numpy())
plot_four(image[0], image[1], mask, pred_plot)
plt.show()
