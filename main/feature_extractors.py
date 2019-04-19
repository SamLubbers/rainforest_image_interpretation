import sys

import numpy as np
from cv2 import BRISK_create
from cv2.xfeatures2d import FREAK_create
from numpy import histogramdd
from skimage.color import rgb2lab, rgb2hsv
from skimage.feature import local_binary_pattern, greycoprops, greycomatrix
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans

sys.path.append("../")
from helpers.img_utils import tif_to_grayscale, tif_to_rgb


class BaseFeatureExtractor(TransformerMixin):
    def __init__(self):
        pass

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
        self.pixels_axis = (1, 2)
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
        self.pixels_axis = (1, 2)
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

    def extract_feature(self, img):
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
            features[i, :] = self.extract_feature(imgs[i, :, :, :])

        return features


class GLCMFeatureExtractor(BaseFeatureExtractor):
    """extracts set of GLCM features from a set of images"""

    def __init__(self):
        self.distances = [1]
        self.directions = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # omnidirectional
        self.glcm_features = ['contrast', 'ASM', 'correlation']
        self.n_features = len(self.glcm_features)
        super().__init__()

    def obtain_property(self, glcm, feature):
        if feature == 'sd':
            return np.std(glcm)
        else:
            return np.mean(greycoprops(glcm, feature))

    def extract_feature(self, img):
        """Obtain glcm feature from tiff image"""
        img_grayscale = tif_to_grayscale(img, as_int=True)
        glcm = greycomatrix(img_grayscale, self.distances, self.directions,
                            symmetric=True, normed=True)
        im_features = np.zeros(self.n_features)
        for i, feature in enumerate(self.glcm_features):
            im_features[i] = self.obtain_property(glcm, feature)
        return im_features

    def fit(self, imgs, y=None):
        self.n_images = np.shape(imgs)[0]
        return self

    def transform(self, imgs, y=None):
        features = np.zeros([self.n_images, self.n_features])
        for i in range(self.n_images):
            features[i, :] = self.extract_feature(imgs[i, :, :, :])

        return features


class GCHFeatureExtractor(BaseFeatureExtractor):
    """extracts global color histogram (GCH) from a set of images"""

    def __init__(self, color_space='rgb'):
        self.n_bins = 8  # number of bins per channel histogram
        self.n_channels = 3
        self.n_features = self.n_bins ** self.n_channels

        self.color_space = color_space
        self.ranges = {'rgb': ((0, 255), (0, 255), (0, 255)),
                       'hsv': ((0, 1), (0, 1), (0, 1)),
                       'lab': ((0, 100), (-128, 127), (-128, 127))}

        self.range = self.ranges[self.color_space]
        super().__init__()

    def preprocess_image(self, img):
        img = tif_to_rgb(img, as_int=True)
        if self.color_space == 'hsv':
            return rgb2hsv(img)
        elif self.color_space == 'lab':
            return rgb2lab(img)
        elif self.color_space == 'rgb':
            return img

    def extract_feature(self, img):
        """Obtain GCH feature from tiff image"""
        img = self.preprocess_image(img)
        GCH, _ = histogramdd(img.reshape(-1, img.shape[-1]),
                             bins=(self.n_bins, self.n_bins, self.n_bins),
                             range=self.range)
        GCH = GCH.flatten() / np.sum(GCH)  # normalize to have L1 norm of 1
        return GCH

    def fit(self, imgs, y=None):
        self.n_images = np.shape(imgs)[0]
        return self

    def transform(self, imgs, y=None):
        features = np.zeros([self.n_images, self.n_features])
        for i in range(self.n_images):
            features[i, :] = self.extract_feature(imgs[i, :, :, :])

        return features


class LocalFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, descriptor='brisk'):
        self.detector = BRISK_create()
        if descriptor == 'brisk':
            self.extractor = self.detector
        elif descriptor == 'freak':
            self.extractor = FREAK_create()

        self.feature_dimension = 64
        super().__init__()

    def extract_feature(self, img):
        img_grayscale = tif_to_grayscale(img, as_int=True)
        keypoints = self.detector.detect(img_grayscale)
        _, descriptors = self.extractor.compute(img_grayscale, keypoints)
        return descriptors

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        self.n_images = np.shape(imgs)[0]
        features = []
        for i in range(self.n_images):
            # append to list rather than numpy.ndarray because number of feature is unknown
            features.append(self.extract_feature(imgs[i, :, :, :]))

        return features


class BoVW(TransformerMixin):
    """Bag of visual words"""

    def __init__(self, n_clusters=1000):
        self.n_clusters = n_clusters  # number of words in the visual bag of words
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def create_histogram(self, img_descriptors):
        clusters = self.kmeans.predict(img_descriptors)
        histogram = np.zeros(self.n_clusters)
        counts = np.unique(clusters, return_counts=True)
        for i, count in zip(counts[0], counts[1]):
            histogram[i] = count
        # normalize histogram
        histogram /= np.sum(histogram)
        return histogram

    def fit(self, descriptors_list, y=None):
        """creates dictionary for the bag of visual words"""
        self.kmeans.fit(np.concatenate(descriptors_list))
        return self

    def transform(self, descriptors_list, y=None):
        """create feature histograms for all descriptors of all images"""
        self.n_images = len(descriptors_list)
        histograms = np.zeros([self.n_images, self.n_clusters])
        for i, descriptor in enumerate(descriptors_list):
            histograms[i, :] = self.create_histogram(descriptor)

        return histograms
