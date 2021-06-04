import torch
import numpy as np
from PIL import Image


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return to_tensor_video(clip)

    def __repr__(self):
        return self.__class__.__name__

def to_tensor_video(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    # _is_tensor_video_clip(clip)
    # if not clip.dtype == torch.uint8:
    #     raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize_video(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)

def normalize_video(clip, mean, std, inplace=True):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

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

class AddNoise(object):

    def __init__(self, scale=0.5):

        self.scale = scale

    def __call__(self, clip):

        return add_noise(clip, scale=self.scale)

def add_noise(clip, scale=0.5):
    return clip + scale * torch.randn_like(clip)


# class NormalizeLatent(object):
#     """
#     Normalize the video clip by mean subtraction and division by standard deviation
#     Args:
#         mean (3-tuple): pixel RGB mean
#         std (3-tuple): pixel RGB standard deviation
#         inplace (boolean): whether do in-place normalization
#     """

#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace

#     def __call__(self, clip):
#         """
#         Args:
#             clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
#         """
#         return (clip - self.mean) / self.std

#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
#             self.mean, self.std, self.inplace)
