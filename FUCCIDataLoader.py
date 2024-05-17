from torch.utils.data import Dataset
import os
import torch
import numpy as np
from aicsimageio import AICSImage
from utils import compute_affinities


class FUCCIDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(
        self,
        root_dir,
        source_channels: tuple,
        target_channels: tuple,
        semantic=None,
        transform=None,
        source_transform=None,
        target_transform=None,
        target_ending=".tif",
        normalize=False,
        source_folder_name="Source",
        target_folder_name="Target",
    ):
        # the directory with all the training samples
        self.root_dir = root_dir
        self.video_files = os.listdir(
            os.path.join(self.root_dir, source_folder_name)
        )  # list the videos
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.semantic = semantic
        self.normalize = normalize

        # transforms applied to raw image
        self.source_transform = source_transform
        self.target_transform = target_transform

        self.open_videos = []
        # we use a list to support videos of varying length
        self.frames_per_video = []

        # same for masks
        self.open_targets = []

        for video_file_base in self.video_files:
            video_file = os.path.join(
                self.root_dir, source_folder_name, video_file_base
            )
            target_file_base = os.path.splitext(video_file_base)[0] + target_ending
            target_file = os.path.join(
                self.root_dir, target_folder_name, target_file_base
            )
            video = AICSImage(video_file)
            target = AICSImage(target_file)
            n_frames_source = video.dims.T
            n_frames_target = target.dims.T
            if not n_frames_target == n_frames_source:
                raise ValueError(
                    f"Video {video_file_base} does not have "
                    "the same frames in target and source"
                )
            self.open_videos.append(video)
            self.open_targets.append(target)
            self.frames_per_video.append(n_frames_source)

    # get the total number of samples
    def __len__(self):
        return sum(self.frames_per_video)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # to determine from which file to read
        # TODO implement check
        video_idx = -1
        frame_idx = -1

        frames_seen = 0
        for i, frames in enumerate(self.frames_per_video):
            frames_seen += frames
            if idx < frames_seen:
                video_idx = i
                frame_idx = idx - (frames_seen - frames)
                break
        # TODO wrap return_dims in functions
        return_dims = "CYX"
        source_frames = self.open_videos[video_idx].get_image_dask_data(
            return_dims, C=self.source_channels, T=frame_idx
        )
        source_frames = source_frames.compute().astype(np.float32)

        # normalize if needed
        if self.normalize:
            for channel_idx in range(source_frames.shape[0]):
                source_frames[channel_idx] = (
                    source_frames[channel_idx] - np.average(source_frames[channel_idx])
                ) / np.std(source_frames[channel_idx])
        source_frames = torch.from_numpy(source_frames)

        target_frames = self.open_targets[video_idx].get_image_dask_data(
            return_dims, C=self.target_channels, T=frame_idx
        )
        target_frames = target_frames.compute()
        # if segmentation task semantic is not None
        if self.semantic:
            # binarize ground truth
            target_frames = target_frames > 0
        elif self.semantic is False:
            # compute affinities
            target_frames = compute_affinities(target_frames, [[0, 1], [1, 0]])
        target_frames = target_frames.astype(np.float32)
        target_frames = torch.from_numpy(target_frames)

        # transform raw image(s)
        if self.source_transform is not None:
            source_frames = self.source_transform(source_frames)
        if self.target_transform is not None:
            target_frames = self.target_transform(target_frames)

        # further transforms on both source and target
        if self.transform is not None:
            # TODO make nicer, hacked for now
            # find batch with non-empty target mask
            batch_found = False
            while not batch_found:
                seed = torch.seed()
                torch.manual_seed(seed)
                selected_source_frames = self.transform(source_frames)
                torch.manual_seed(seed)
                selected_target_frames = self.transform(target_frames)
                if np.greater(selected_target_frames.mean(), 0.0):
                    batch_found = True
            return selected_source_frames, selected_target_frames
        else:
            return source_frames, target_frames


'''
def load_nd2_file(nd2_file):
    """Load the file (TODO flat field correction?)"""

    return loaded_file
'''
