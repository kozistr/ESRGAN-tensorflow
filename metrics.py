import cv2
import numpy as np


def calc_psnr(img1, img2):
    """ calculate PSNR
    :param img1: np.array. expected [0, 255] scale
    :param img2: np.array. expected [0, 255] scale
    :return: np.float. PSNR value
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean(np.subtract(img1, img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def ssim(img1, img2):
    """ SSIM formula
    :param img1: np.array.
    :param img2: np.array.
    :return: np.float. SSIM value
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    _border = 5  # valid
    _k1 = .01
    _k2 = .03
    _c1 = (_k1 * 255.) ** 2
    _c2 = (_k2 * 255.) ** 2
    _kernel = cv2.getGaussianKernel(ksize=11, sigma=1.5)
    _window = np.outer(_kernel, _kernel.T)

    _mu1 = cv2.filter2D(img1, -1, _window)[_border:-_border, _border:-_border]
    _mu2 = cv2.filter2D(img2, -1, _window)[_border:-_border, _border:-_border]

    _mu1_sq = _mu1 ** 2
    _mu2_sq = _mu2 ** 2
    _mu1_mu2 = _mu1 * _mu2

    _sigma1_sq = cv2.filter2D(img1 ** 2, -1, _window)[_border:-_border, _border:-_border] - _mu1_sq
    _sigma2_sq = cv2.filter2D(img2 ** 2, -1, _window)[_border:-_border, _border:-_border] - _mu2_sq
    _sigma1_sigma2 = cv2.filter2D(img1 * img2, -1, _window)[_border:-_border, _border:-_border] - _mu1_mu2

    _ssim = ((2. * _mu1_mu2 + _c1) * (2. * _sigma1_sigma2 + _c2)) / \
            ((_mu1_sq + _mu2_sq + _c1) * (_sigma1_sq + _sigma2_sq + _c2))
    return np.mean(_ssim)


def calc_ssim(img1, img2):
    """ calculate SSIM
    :param img1: np.array. expected [0, 255] scale
    :param img2: np.array. expected [0, 255] scale
    :return: np.float. PSNR value
    """
    assert img1.shape == img2.shape

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[-1] == 1:  # in case of 'grayscale'
            return ssim(np.squeeze(img1), np.squeeze(img2))
        elif img1.shape[-1] == 3:  # in case of 'rgb'
            _ssims = np.asarray([ssim(img1, img2) for _ in range(img1.shape[-1])])
            return np.mean(_ssims)
        else:
            raise ValueError("[-] not supported channel size ({})".format(img1.shape[-1]))
    else:
        raise ValueError("[-] not supported dimension ({})".format(img1.ndim))
