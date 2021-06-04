import torch
import torchvision
from torchvision import io
from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader

# from utils import *
# from utils.transforms import *

import os
import time
import json
import pickle
import h5py
import random
import cv2


class BehavioralVideo(Dataset):
    """
    Assuming video format = cat[view1, view2]
    """
    def __init__(self, frames_per_clip, frame_rate=4, resolution=400, anchor=600, interval=[750,1150], grayscale=True, 
                    mod='train', transform=None, rand_crop=False, jitter=True, return_first_frame=False, first_frame_jitter_range=15):
        self.mod = mod
        self.step = frame_rate
        self.fpc = frames_per_clip
        self.size = resolution
        self.anchor = anchor
        self.interval = interval
        self.rand_crop = rand_crop
        self.jitter = jitter
        self.return_first_frame = return_first_frame
        self.first_frame_jitter_range  = first_frame_jitter_range
        self.grayscale = grayscale

        with open("./configs/split.json", "rb") as f:
            self.trials = json.load(f)[mod]
        self.transforms = torchvision.transforms.Compose([transform])
        print('Total {} trials: '.format(mod), len(self.trials))

    def get_clip(self, trial_dir, rand_crop=False, jitter=False):

        # read from raw videos
        vidcap = cv2.VideoCapture(trial_dir)
        raw_video_flow, success = [], True
        while success: 
            success, frame = vidcap.read()
            raw_video_flow.append(frame)
        raw_video_flow = raw_video_flow[:-1]
        if self.grayscale:
            raw_video_flow = [frame[:, :, 0] for frame in raw_video_flow]

        # random crop
        rand = random.randint(-self.step, self.step) if jitter else 0
        if rand_crop:
            start = self.interval[0] + random.randint(0, self.interval[1]-self.interval[0]-self.fpc*self.step)
        else:
            start = self.interval[0]
        start, end = rand+start, rand+start+self.fpc*self.step
        clip_indices = range(start, end, self.step)
        clip = [to_tensor(raw_video_flow[idx]) for idx in clip_indices]
        clip = torch.stack(clip, dim=0)
        
        # read context frames
        if self.return_first_frame:
            first_frame_jitter = random.randint(-self.first_frame_jitter_range, self.first_frame_jitter_range) if jitter else 0
            anchor = max(0, self.anchor+first_frame_jitter)
            first_frame = to_tensor(raw_video_flow[anchor])
            clip = torch.cat([first_frame.unsqueeze(0), clip], dim=0)
        
        return clip

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):

        clip = self.get_clip(self.trials[idx], rand_crop=self.rand_crop, jitter=self.jitter)
        front, side = clip[:, :, :, self.size:], clip[:, :, :, :self.size]

        if self.transforms is not None:
            front, side = self.transforms(front), self.transforms(side)

        return front, side
    

class ResizeVideo(object):
    """
    Resize the tensors
    Args:
        target_size: int or tuple
    """
    def __init__(self, target_size, interpolation_mode='bilinear'):
        assert isinstance(target_size, int) or (isinstance(target_size, Iterable) and len(target_size) == 2)
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W) / (T, C, H, W)
        """
        return resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__

def resize(clip, target_size, interpolation_mode):
    # assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=True
    )
