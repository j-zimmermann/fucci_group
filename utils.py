"""Taken from https://github.com/dl4mia"""

import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.color import gray2rgb, rgb2gray
from skimage.graph import rag_boundary, merge_hierarchical, show_rag
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_dist_between_points):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, n = label(maxima)

    return seeds, n


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


def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """Function to evaluate a segmentation."""

    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}

    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)

    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # create iou table
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou

    # remove background
    iouMat = iouMat[1:, 1:]

    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = tp / (tp + fp + fn)

    return precision, recall, accuracy


def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )

    return segmentation


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count,
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def get_labels_from_prediction(prediction: torch.Tensor, id_offset, min_seed_distance):
    # convert to cpu
    prediction = np.squeeze(prediction.cpu().detach().numpy())
    threshold = threshold_otsu(prediction)

    # get boundary mask
    inner_mask = 0.5 * (prediction[0] + prediction[1]) > threshold

    boundary_distances = distance_transform_edt(inner_mask)

    pred_labels = watershed_from_boundary_distance(
        boundary_distances,
        inner_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance,
    )
    """
    pred_edges = sobel(pred_labels)

    plt.imshow(pred_labels)
    plt.show()
    plt.imshow(pred_edges)
    plt.show()


    graph = rag_boundary(pred_labels, pred_edges)

    show_rag(pred_labels, graph, gray2rgb(prediction[0] + prediction[1]), img_cmap="gray")
    plt.title('Initial RAG')
    plt.show()

    
    pred_labels = merge_hierarchical(
        pred_labels,
        graph,
        thresh=0.08,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=weight_boundary,
    )

    print(pred_labels.shape)
    show_rag(pred_labels, graph, gray2rgb(prediction[0] + prediction[1]), img_cmap="gray")
    plt.title('RAG after hierarchical merging')
    plt.show()

    pred_labels = rgb2gray(pred_labels)

    plt.imshow(pred_labels)
    plt.title('Final segmentation')

    plt.show()
    """

    return pred_labels


def validation_instance_seg(
    model,
    loader,
    loss_function,
    metric=False,
    device=None,
    id_offset=0,
    min_seed_distance=3,
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
    val_metric = np.zeros(3)

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
            if metric:
                gt_labels = get_labels_from_prediction(y, id_offset, min_seed_distance)
                pred_labels = get_labels_from_prediction(
                    prediction, id_offset, min_seed_distance
                )
                metric_values = evaluate(gt_labels, pred_labels)
                for i, _ in enumerate(metric_values):
                    val_metric[i] = metric_values[i]

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric = val_metric / len(loader)

    # return nan if not computed
    if not metric:
        val_metric[:] = np.nan

    return val_loss, val_metric
