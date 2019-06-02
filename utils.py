import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm


class ImageDataLoader:
    def __init__(self,
                 image_shape: tuple = (128, 128),
                 channels: int = 3):
        self.image_shape = image_shape
        self.channels = channels

    def pre_processing(self, fn):
        lr = tf.read_file(fn[0])
        lr = tf.image.decode_png(lr, channels=self.channels)
        lr = tf.cast(lr, dtype=tf.float32) / 255.
        lr = tf.reshape(lr, self.image_shape + (self.channels,))

        hr = tf.read_file(fn[1])
        hr = tf.image.decode_png(hr, channels=self.channels)
        hr = tf.cast(hr, dtype=tf.float32) / 255.
        hr = tf.reshape(hr, self.image_shape + (self.channels,))
        return lr, hr


def bgr2ycbcr(img: np.array, only_y: bool = True):
    """ bgr image to ycbcr image,
    inspired by https://github.com/xinntao/BasicSR/blob/master/metrics/calculate_PSNR_SSIM.py
    :param img: np.array. expected types, uint8 & float
        uint8 : [0, 255]
        float : [0, 1]
    :param only_y: bool. return only y channel
    :return: np.array.
    """
    _dtype = img.dtype
    if not _dtype == np.uint8:
        img *= 255.

    img.astype(np.float32)

    if only_y:
        rlt = np.dot(img,
                     [24.966, 128.553, 65.481]) / 255. + 16.
    else:
        rlt = np.matmul(img, [
            [24.966, 112., -18.214],
            [128.553, -74.203, -93.786],
            [65.481, -37.797, 112.]
        ]) / 255. + [16, 128, 128]

    if _dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(_dtype)


def load_image_from_file(fn: str, mode: str = 'rgb',
                         normalize: bool = False, norm_scale: str = "0,1"):
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


def load_images_from_path(path: str, ext: str = "png", mode: str = 'rgb',
                          normalize: bool = False, norm_scale: str = "0,1"):
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
