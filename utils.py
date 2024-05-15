"""Taken from https://github.com/dl4mia"""

import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np


class DiceCoefficient(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        union = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / union.clamp(min=self.eps)


class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction * weights, target * weights
        )


def compute_affinities(seg: np.ndarray, nhood: list):
    seg = np.squeeze(seg)
    nhood = np.array(nhood)

    shape = seg.shape
    n_edges = nhood.shape[0]
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32)

    for e in range(n_edges):
        affinity[
            e,
            max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
        ] = (
            (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                == seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
            )
            * (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                > 0
            )
            * (
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
                > 0
            )
        )

    return affinity


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            raise NotImplementedError("Prediction shape differs from expected shape!")
            # TODO fix, crop is not defined
            # y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        if isinstance(loss_function, WeightedMSELoss):
            # TODO currently assumes single channel
            classes, counts = torch.unique(y, return_counts=True)
            if len(classes) != 2:
                raise RuntimeError("Too many labels for weighting")
            weights = torch.zeros(y.shape)
            weights = weights.to(device)
            # go through classes and assign weights
            for class_idx, count in zip(classes, counts):
                weights[y == class_idx] = count / torch.numel(y)
            weights = 1.0 / weights
            loss = loss_function(prediction, y, weights)
        else:
            loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


def validate(
    model,
    loader,
    loss_function,
    metric,
    device=None,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            val_loss += loss_function(prediction, y).item()
            # TODO introduce variable for segmentation threshold
            val_metric += metric(prediction > 0.5, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    return val_loss, val_metric


def plot_three(
    image: np.ndarray, image_ch2: np.ndarray, pred: np.ndarray, label: str = "Target"
):
    """
    Helper function to plot an image, the auxiliary (image_ch2)
    representation of the target and the model prediction.
    """
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_xlabel("Image", fontsize=20)
    plt.imshow(image, cmap="magma")
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_xlabel(label, fontsize=20)
    plt.imshow(image_ch2, cmap="magma")
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.set_xlabel("Prediction", fontsize=20)
    t = plt.imshow(pred, cmap="magma")
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3]]  # remove the yticks
    plt.tight_layout()
    plt.show()


def plot_four(
    image: np.ndarray,
    image_ch2: np.ndarray,
    pred: np.ndarray,
    seg: np.ndarray,
    cmap: str = "nipy_spectral",
):
    """
    Helper function to plot an image, the auxiliary (image_ch2)
    representation of the target, the model prediction and the predicted segmentation mask.
    """

    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    t = ax1.imshow(image, cmap="gray")  # show the image
    ax1.set_xlabel("CH1", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(spec[0, 1])
    t = ax2.imshow(image_ch2, cmap="gray")  # show the masks
    ax2.set_xlabel("CH2", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(spec[0, 2])
    t = ax3.imshow(pred)
    ax3.set_xlabel("GT", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    ax4 = fig.add_subplot(spec[0, 3])
    t = ax4.imshow(seg)
    ax4.set_xlabel("Pred.", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)

    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the yticks
    plt.tight_layout()


def plot_five(
    image: np.ndarray,
    image_ch2: np.ndarray,
    image_ch3: np.ndarray,
    pred: np.ndarray,
    seg: np.ndarray,
    cmap: str = "nipy_spectral",
):
    """
    Helper function to plot an image, the auxiliary (image_ch2)
    representation of the target, the model prediction and the predicted segmentation mask.
    """

    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=5, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.imshow(image, cmap="gray")  # show the image
    ax1.set_xlabel("CH1", fontsize=20)
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.imshow(image_ch2, cmap="gray")  # show the masks
    ax2.set_xlabel("CH2", fontsize=20)
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.imshow(image_ch3, cmap="gray")  # show the masks
    ax3.set_xlabel("CH3", fontsize=20)
    ax4 = fig.add_subplot(spec[0, 3])
    t = ax4.imshow(pred, interpolation=None)
    ax4.set_xlabel("GT", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    ax5 = fig.add_subplot(spec[0, 4])
    t = ax5.imshow(seg, interpolation=None)
    ax5.set_xlabel("Pred.", fontsize=20)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3, ax4, ax5]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3, ax4, ax5]]  # remove the yticks
    plt.tight_layout()
    plt.show()
