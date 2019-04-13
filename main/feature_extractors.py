import numpy as np
from skimage.io import concatenate_images
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


def extract_features(imgs_collection, feature_extractor, batch_size=1000):
    """
    Extracts from imgs_collection the set of features specified in feature_extractor.
    Extracts the features in batches because it is unviable to load all images at once into memory

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features
    feature_extractor: sklearn transformer
                       transformers that extract features from images
    batch_size: number of images to extract features from at each iteration
                a deafult value of 1000 loads approx. 0.5 GB into memory for this dataset

    Returns
    -------
    features: numpy.ndarray (np.float64)
              set of features extracted from the image,
              with one row for each image and one column for each feature
    """

    n_images = len(imgs_collection)
    # get number of total features
    features_im0 = feature_extractor.fit_transform(concatenate_images(imgs_collection[:1]))
    n_features = np.shape(features_im0)[1]
    # create array for features
    features = np.zeros([n_images, n_features])
    features[0, :] = features_im0
    for i in range(1, n_images, batch_size):
        imgs_batch = concatenate_images(imgs_collection[i:i + batch_size])
        # casting is required for feature extraction!
        # else the default uint16 produces errors in the computations
        imgs_batch = imgs_batch.astype('float64', casting='safe')
        features_batch = feature_extractor.fit_transform(imgs_batch)
        features[i:i + batch_size, :] = features_batch
    return features
