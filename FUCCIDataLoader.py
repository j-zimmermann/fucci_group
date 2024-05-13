from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from aicsimageio import AICSImage


class FUCCIDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(
        self,
        root_dir,
        source_channels: tuple,
        target_channels: tuple,
        transform=None,
        img_transform=None,
        target_ending=".tif"
    ):
        self.root_dir = os.path.join(
            "/group/dl4miacourse/projects/FUCCI", root_dir
        )  # the directory with all the training samples
        self.video_files = os.listdir(
            os.path.join(self.root_dir, "Source")
        )  # list the videos
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.source_channels = source_channels
        self.target_channels = target_channels

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.0], [1.0]),  # 0 = mean and 1 = variance
            ]
        )

        self.open_videos = []
        # we use a list to support videos of varying length
        self.frames_per_video = []

        # same for masks
        self.open_targets = []

        for video_file_base in self.video_files:
            video_file = os.path.join(self.root_dir, "Source", video_file_base)
            target_file_base = os.path.splitext(video_file_base)[0] + target_ending
            target_file = os.path.join(self.root_dir, "Target", target_file_base)
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
        return_dims = "YX"
        if isinstance(self.source_channels, tuple):
            if len(self.source_channels) > 1:
                return_dims = "CYX"
        source_frames = self.open_videos[video_idx].get_image_data(
            return_dims, C=self.source_channels, T=frame_idx
        )

        if isinstance(self.target_channels, tuple):
            if len(self.target_channels) > 1:
                return_dims = "CYX"

        target_frames = self.open_targets[video_idx].get_image_data(
            return_dims, C=self.target_channels, T=frame_idx
        )
        if self.transform is not None:
            seed = torch.seed()
            torch.manual_seed(seed)
            source_frames = self.transform(source_frames)
            torch.manual_seed(seed)
            target_frames = self.transform(target_frames)
        if self.img_transform is not None:
            source_frames = self.img_transform(source_frames)
        return source_frames, target_frames


'''
def load_nd2_file(nd2_file):
    """Load the file (TODO flat field correction?)"""

    return loaded_file
'''
