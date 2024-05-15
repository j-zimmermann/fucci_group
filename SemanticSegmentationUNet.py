from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, validate, DiceCoefficient
from tqdm import tqdm
from torchvision import transforms
import pandas as pd

n_epochs = 250
model_directory = "semantic_2_channel_model"
if not os.path.isdir(model_directory):
    os.mkdir(model_directory)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()

source_channels = (0, 3)
target_channels = 0

print("Loading data")
# TODO add more transforms
train_data = FUCCIDataset(
    root_dir="JuliusTrain",
    transform=transforms.RandomCrop(256),
    source_transform=None,
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True,
)

validate_data = FUCCIDataset(
    root_dir="JuliusValidate",
    transform=transforms.RandomCrop(256),
    source_transform=None,
    normalize=True,
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    semantic=True,
)

train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)
validation_loader = DataLoader(validate_data, batch_size=5, shuffle=True, num_workers=8)

unet = UNet(
    depth=4,
    in_channels=len(source_channels),
    out_channels=1,
    final_activation="Sigmoid",  # needed for masks
    num_fmaps=16,
)
unet.to(device)


loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
metric = DiceCoefficient()


training_log = {}
training_log["epoch"] = []
training_log["validation_metric"] = []
training_log["validation_loss"] = []

print("Start training")
for epoch in tqdm(range(n_epochs)):
    train(unet, train_loader, optimizer, loss, epoch, device=device)
    val_loss, val_metric = validate(
        unet, validation_loader, loss, metric, device=device
    )
    training_log["epoch"].append(epoch)
    training_log["validation_metric"].append(val_metric)
    training_log["validation_loss"].append(val_loss)

    # save snapshot
    if epoch % 5 == 0:
        torch.save(
            unet.state_dict(), os.path.join(model_directory, "training_state.pt")
        )


print("Save training log")
df = pd.DataFrame(training_log)
df.to_csv(os.path.join(model_directory, "training_log.csv"), index=False)
print("Done")
