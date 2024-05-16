from FUCCIDataLoader import FUCCIDataset
import torch
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils import train, WeightedMSELoss, validation_instance_seg
from tqdm import tqdm
from torchvision import transforms
import pandas as pd

n_epochs = 150

model_directory = "instance_2_channel_model"
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
    source_channels=source_channels,
    target_channels=target_channels,
    target_ending=".tif",
    normalize=True,
    semantic=False,
)

train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)

unet = UNet(
    depth=4,
    in_channels=len(source_channels),
    out_channels=2,
    final_activation="Sigmoid",  # needed for masks
    num_fmaps=16,
)
unet.to(device)


loss = WeightedMSELoss()
optimizer = torch.optim.Adam(unet.parameters())

training_log = {}
training_log["epoch"] = []
training_log["validation_recall"] = []
training_log["validation_precision"] = []
training_log["validation_accuracy"] = []
training_log["validation_loss"] = []

print("Start training")
for epoch in tqdm(range(n_epochs)):
    train(unet, train_loader, optimizer, loss, epoch, device=device)
    if epoch % 5 == 0:
        torch.save(
            unet.state_dict(), os.path.join(model_directory, "training_state.pt")
        )

    val_loss, metrics = validation_instance_seg(
        unet, train_loader, torch.nn.MSELoss(), metric=epoch % 5, device=device
    )
    precision, recall, accuracy = metrics
    training_log["epoch"].append(epoch)
    training_log["validation_loss"].append(val_loss)
    training_log["validation_recall"].append(recall)
    training_log["validation_precision"].append(precision)
    training_log["validation_accuracy"].append(accuracy)

print("Save training log")
df = pd.DataFrame(training_log)
df.to_csv(os.path.join(model_directory, "training_log.csv"), index=False)
print("Done")
