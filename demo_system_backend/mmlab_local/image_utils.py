# Extracted from mmcv.image.geometric and mmcv.image.photometric.
# Self-contained: only depends on numpy and cv2.

from typing import Optional, Tuple, Union

import cv2
import numpy as np


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size: (w, h).
        scale: Scaling factor or maximum size (short_edge, long_edge).
        return_scale: Whether to return the scaling factor.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    return new_size


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size (w, h) using cv2."""
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    w_scale = size[0] / w
    h_scale = size[1] / h
    return resized_img, w_scale, h_scale


def imnormalize_(img: np.ndarray, mean, std, to_rgb: bool = True):
    """Inplace normalize an image with mean and std."""
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img
