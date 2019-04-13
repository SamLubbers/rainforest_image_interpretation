import numpy as np
from sklearn.base import TransformerMixin


class BaseFeatureExtractor(TransformerMixin):
    def __init__(self):
        self.pixels_axis = (1, 2)

    def fit(self, imgs, y=None):
        raise NotImplementedError

    def transform(self, imgs, y=None):
        raise NotImplementedError


class SpectralFeatureExtractor(BaseFeatureExtractor):
    """
    extracts mean and standard deviation of every color channel in the image (RGB)
    and the brightness, where brightness is defined as the mean of all color channels

    Parameters
    ----------
    imgs : numpy.ndarray (np.float)
           set of images, each with 4 channels (B, G, R, NIR)
    """

    def __init__(self):
        super().__init__()

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        imgs = imgs[:, :, :, :3]  # extract color channels
        rgb_means = np.mean(imgs, axis=self.pixels_axis)
        brightness = np.mean(rgb_means, axis=1)
        brightness = np.reshape(brightness, (-1, 1))
        rgb_sds = np.std(imgs, axis=self.pixels_axis)

        return np.concatenate((rgb_means, brightness, rgb_sds), axis=1)


class NDVIFeatureExtractor(BaseFeatureExtractor):
    """
    extracts normalized difference vegatation index from multispectral image

    Parameters
    ----------
    imgs : numpy.ndarray (np.float)
           set of images, each with 4 channels (B, G, R, NIR)
    """

    def __init__(self):
        super().__init__()

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        red = imgs[:, :, :, 2]
        nir = imgs[:, :, :, 3]

        ndvi = np.divide(nir - red, nir + red)

        ndvi_means = np.mean(ndvi, axis=self.pixels_axis)
        ndvi_sds = np.std(ndvi, axis=self.pixels_axis)
        ndvi_means = np.reshape(ndvi_means, (-1, 1))
        ndvi_sds = np.reshape(ndvi_sds, (-1, 1))
        return np.concatenate((ndvi_means, ndvi_sds), axis=1)
