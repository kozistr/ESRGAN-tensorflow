import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm


def load_image_from_file(fn, mode='rgb', normalize=False, norm_scale="0,1"):
    """ loading an image from file name
    :param fn: str. file name
    :param mode: str. mode to load image
    :param normalize: bool.
    :param norm_scale: str. range of image pixel value to normalize
    :return: np.array. rgb image
    """
    assert mode in ('rgb', 'bgr', 'grayscale')
    assert norm_scale in ("0,1", "-1,1")

    img = cv2.imread(fn, cv2.IMREAD_COLOR if not mode == "grayscale" else cv2.IMREAD_GRAYSCALE)

    if mode == "rgb":
        img = img[::-1]

    if normalize:
        if norm_scale == "0,1":
            img /= 255.
        else:
            img = (img / 127.5) - 1.
    return img


def load_images_from_path(path, ext="png", mode='rgb', normalize=False, norm_scale="0,1"):
    """ loading an image from file name
    :param path: str. file path
    :param ext: str. extension
    :param mode: str. mode to load image
    :param normalize: bool.
    :param norm_scale: str. range of image pixel value to normalize
    :return: list of np.array.
    """
    assert mode in ('rgb', 'grayscale')
    assert norm_scale in ("0,1", "-1,1")

    _files = glob(os.path.join(path, "*.{}".format(ext)))

    images = np.asarray([load_image_from_file(_file, mode, normalize, norm_scale)
                         for _file in tqdm(_files)])
    return images


if __name__ == "__main__":
    pass
