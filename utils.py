import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm

seed: int = 13371337

np.random.seed(seed)
tf.set_random_seed(seed)


class ImageDataLoader:
    def __init__(self,
                 patch_shape: tuple = (128, 128),
                 channels: int = 3,
                 patch_size: int = 16):
        self.patch_shape = patch_shape
        self.channels = channels
        self.patch_size = patch_size
        self.scale = int(np.sqrt(self.patch_size))

        self.lr_patch_shape = (
            self.patch_shape[0],
            self.patch_shape[1])
        self.hr_patch_shape = (
            self.patch_shape[0] * self.scale,
            self.patch_shape[1] * self.scale
        )

    def random_crop(self, x_lr, x_hr):
        x_hr_shape = x_hr.get_shape().as_list()

        rand_lr_w = (np.random.randint(0, x_hr_shape[0] - self.hr_patch_shape[0])
                     // self.scale)
        rand_lr_h = (np.random.randint(0, x_hr_shape[1] - self.hr_patch_shape[1])
                     // self.scale)
        rand_hr_w = rand_lr_w * self.scale
        rand_hr_h = rand_lr_h * self.scale

        x_lr = x_lr[rand_lr_w:rand_lr_w + self.lr_patch_shape[0], rand_lr_h:rand_lr_h + self.lr_patch_shape[1], :]
        x_hr = x_hr[rand_hr_w:rand_hr_w + self.hr_patch_shape[0], rand_hr_h:rand_hr_h + self.hr_patch_shape[1], :]
        return x_lr, x_hr

    def pre_processing(self, fn):
        lr = tf.read_file(fn[0])
        lr = tf.image.decode_png(lr, channels=self.channels)
        lr = tf.cast(lr, dtype=tf.float32) / 255.

        hr = tf.read_file(fn[1])
        hr = tf.image.decode_png(hr, channels=self.channels)
        hr = tf.cast(hr, dtype=tf.float32) / 255.

        # random crop
        lr, hr = self.random_crop(lr, hr)

        # augmentations
        if np.random.randint(0, 2) == 0:
            lr = tf.image.flip_up_down(lr)
            hr = tf.image.flip_up_down(hr)

        if np.random.randint(0, 2) == 0:
            lr = tf.image.rot90(lr)
            hr = tf.image.rot90(hr)

        # split into patches
        lr_patches = tf.image.extract_image_patches(
            images=tf.expand_dims(lr, axis=0),
            ksizes=(1,) + self.lr_patch_shape + (1,),
            strides=(1,) + self.lr_patch_shape + (1,),
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        lr_patches = tf.reshape(lr_patches,
                                (-1,) + self.lr_patch_shape + (self.channels,))

        hr_patches = tf.image.extract_image_patches(
            images=tf.expand_dims(hr, axis=0),
            ksizes=(1,) + self.hr_patch_shape + (1,),
            strides=(1,) + self.hr_patch_shape + (1,),
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        hr_patches = tf.reshape(hr_patches,
                                (-1,) + self.hr_patch_shape + (self.channels,))

        return lr_patches, hr_patches


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
