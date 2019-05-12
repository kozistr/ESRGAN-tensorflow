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
