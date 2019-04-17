import sys

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.base import TransformerMixin

sys.path.append("../")
from helpers.img_utils import tif_to_grayscale


class BaseFeatureExtractor(TransformerMixin):
    def __init__(self):
        self.pixels_axis = (1, 2)

    def fit(self, imgs, y=None):
        raise NotImplementedError

    def transform(self, imgs, y=None):
        raise NotImplementedError


class ColorChannelsFeatureExtractor(BaseFeatureExtractor):
    """
    extracts mean and standard deviation of every color channel in the image (RGB)
    and the brightness, where brightness is defined as the mean of all color channels

    Parameters
    ----------
    imgs : numpy.ndarray (np.int | np.float)
           set of images, each with 4 channels (B, G, R, NIR)
    """

    def __init__(self):
        super().__init__()

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        imgs = imgs[:, :, :, :3]  # extract color channels
        rgb_means = np.mean(imgs, axis=self.pixels_axis)
        rgb_sds = np.std(imgs, axis=self.pixels_axis)
        brightness = np.mean(rgb_means, axis=1)
        brightness = np.reshape(brightness, (-1, 1))

        return np.concatenate((rgb_means, rgb_sds, brightness), axis=1)


class NDVIFeatureExtractor(BaseFeatureExtractor):
    """
    extracts normalized difference vegatation index from multispectral image

    Parameters
    ----------
    imgs : numpy.ndarray (np.int | np.float)
           set of images, each with 4 channels (B, G, R, NIR)
    """

    def __init__(self):
        super().__init__()

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        # casting is required to obtain negative NDVI values. Else the computations produce errors for uint
        imgs = imgs.astype('float64', casting='safe')

        red = imgs[:, :, :, 2]
        nir = imgs[:, :, :, 3]

        ndvi = np.divide(nir - red, nir + red)

        ndvi_means = np.mean(ndvi, axis=self.pixels_axis)
        ndvi_sds = np.std(ndvi, axis=self.pixels_axis)
        ndvi_means = np.reshape(ndvi_means, (-1, 1))
        ndvi_sds = np.reshape(ndvi_sds, (-1, 1))
        return np.concatenate((ndvi_means, ndvi_sds), axis=1)


class LBPFeatureExtractor(BaseFeatureExtractor):
    """extracts LBP features of a set of images using the 'uniform' method"""

    def __init__(self, r):
        """
        Parameters
        ----------
        r: int
           radius of LBP
        """
        self.r = r
        self.p = 8 * r
        self.n_bins = self.p + 2
        super().__init__()

    def tiff_lbp(self, img):
        """Obtain lbp feature from tiff image"""
        img_grayscale = tif_to_grayscale(img)
        lbp = local_binary_pattern(img_grayscale, self.p, self.r, method='uniform')
        lbp_feature = np.histogram(lbp, bins=self.n_bins, range=(0, self.n_bins))[0]
        return lbp_feature

    def fit(self, imgs, y=None):
        self.n_images = np.shape(imgs)[0]
        return self

    def transform(self, imgs, y=None):
        features = np.zeros([self.n_images, self.n_bins])
        for i in range(self.n_images):
            features[i, :] = self.tiff_lbp(imgs[i, :, :, :])

        return features
