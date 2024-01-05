# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import Dict, Tuple

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

def events2frame(events: np.array,
                 size: Tuple[Height, Width]) -> np.array:
    
    frame = np.zeros((size[0], size[1], 2), dtype=np.uint8)
                  
    valid = (events['x'] >= 0) & (events['x'] < size[1]) & \
            (events['y'] >= 0) & (events['y'] < size[0])
    events = events[valid]
    frame[events['y'][events['p'] == 1],
            events['x'][events['p'] == 1], 0] = 1
    frame[events['y'][events['p'] == 0],
            events['x'][events['p'] == 0], 1] = 1
    
    return frame

def resize_events_frame(data: Dict,
                        size: Tuple[Height, Width]) -> np.array:
    height, width = data['size']['height'], data['size']['width']
    events = data['events']
    events['y'] = (events['y'] * (size[0] / height)).astype(int)
    events['x'] = (events['x'] * (size[1] / width)).astype(int)
    return events2frame(events, size)


def fliplr_events(events: np.array) -> np.array:
    return np.fliplr(events)
