# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple

import numpy as np
from PIL.Image import Image, Transpose

Width = int
Height = int

"""Dataset manipulation utility module."""


def flip_lr(image: Image) -> Image:
    """Flip a PIL image left-right.

    Parameters
    ----------
    image : Image
        Input image.

    Returns
    -------
    Image
        Flipped image.
    """
    return Image.transpose(image, Transpose.FLIP_LEFT_RIGHT)


def flip_ud(image: Image) -> Image:
    """Flip a PIL image up-down.

    Parameters
    ----------
    image : Image
        Input image.

    Returns
    -------
    Image
        Flipped image.
    """
    return Image.transpose(image, Transpose.FLIP_TOP_BOTTOM)

def resize_events_frame_channel(events: np.array,
                                size: Tuple[Height, Width]) -> np.array:
    height = events.shape[0]
    width = events.shape[1]
    # need to match array
    pixels_keep = [(height // size[0]) * size[0], (width // size[1]) * size[1]]
    events = events[:pixels_keep[0], :pixels_keep[1]]
    shape = (size[0], height // size[0], size[1], width // size[1])
    img = events.reshape(shape)
    return img[:, 0, :, 0]

def resize_events_frame(events: np.array,
                        size: Tuple[Height, Width]) -> np.array:
    
    # img = np.zeros((size[0], size[1], 2))
    # img[:, :, 0] = resize_events_frame_channel(events[:, :, 0], size)
    # img[:, :, 1] = resize_events_frame_channel(events[:, :, 1], size)
    
    
    height = events.shape[0]
    width = events.shape[1]

    img = np.asarray(
        [[events[int(height * rows / size[0])][int(width * cols / size[1])]
          for cols in range(size[1])] for rows in range(size[0])])
    return img


def fliplr_events(events: np.array) -> np.array:
    return np.fliplr(events)
