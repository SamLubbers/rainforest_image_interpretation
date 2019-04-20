import os
import sys
from time import time

import numpy as np
import pandas as pd
from skimage.io import concatenate_images
from sklearn.ensemble import RandomForestClassifier
from itertools import chain
from sklearn.pipeline import FeatureUnion, Pipeline

from feature_extractors import BoVW

sys.path.append("../")
from config import DATASETS_PATH, TRAIN_PATH, VALIDATION_PATH
from helpers.img_utils import load_image_collection
from helpers.data import extract_label_values
from helpers.evaluation import evaluate_performance_validation
from multiprocessing import Pool, cpu_count


def extract_features_batch(imgs_collection_batch, feature_extractor, mode):
    """extracts features of a collection of images
    requires as separate function for parallel processing
    """

    imgs_batch = concatenate_images(imgs_collection_batch)
    if mode == 'train':
        feature_extractor.fit(imgs_batch)

    return feature_extractor.transform(imgs_batch)


def extract_global_features(imgs_collection, feature_extractor, mode, batch_size=200):
    """
    Extracts from imgs_collection the set of global features specified in feature_extractor.
    Extracts the features in batches because it is unviable to load all images at once into memory

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features

    feature_extractor: sklearn transformer | sklearn Pipeline | sklearn FeatureUnion
                       transformers that extract features from images

    mode: 'train' / 'validate'. Determines whether the transformers are fitted to the data or not.

    batch_size: number of images to extract features from in each process
                total memory used at a given time is cpu_cout * memory occupied
                by batch_size number of images

    Returns
    -------
    features: numpy.ndarray (np.float64)
              set of features extracted from the image,
              with one row for each image and one column for each feature
    """

    n_images = len(imgs_collection)

    imgs_collection_batches = [imgs_collection[i:i + batch_size]
                               for i in range(0, n_images, batch_size)]

    with Pool(cpu_count()) as pool:
        batch_features = [pool.apply(extract_features_batch, args=(ims, feature_extractor, mode))
                          for ims in imgs_collection_batches]

    return np.concatenate(batch_features)


def extract_local_feature_descriptors(imgs_collection, local_feature_extractor, mode, batch_size=1000):
    n_images = len(imgs_collection)
    descriptors = []
    for i in range(0, n_images, batch_size):
        imgs_batch = concatenate_images(imgs_collection[i:i + batch_size])
        if mode == 'train':
            local_feature_extractor.fit(imgs_batch)

        batch_descriptors = local_feature_extractor.transform(imgs_batch)
        descriptors.append(batch_descriptors)

    return list(chain.from_iterable(descriptors))


def extract_local_features(imgs_collection, local_feature_pipeline, mode):
    """

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features

    local_feature_pipeline : sklearn Pipeline
                             Pipeline consisting of two transformers. The first one extracts local features
                             and the second one converts those features to bag of words representation

    mode: 'train' / 'validate'. Determines whether the transformers are fitted to the data or not.

    batch_size: number of images to extract features at each step.
                Default value equates to about 0.5GB of memory
    """

    local_feature_extractor, bovw = local_feature_pipeline.named_steps.values()

    # extract local features
    # feature extraction is done sequentially because cv2 objects do not work with multiprocessing library

    local_feature_descriptors = extract_local_feature_descriptors(imgs_collection,
                                                                  local_feature_extractor,
                                                                  mode)

    if mode == 'train':
        # create dictionary
        bovw.fit(local_feature_descriptors)

    # convert local features to bovw representation
    return bovw.transform(local_feature_descriptors)


def extract_features(imgs_collection, feature_extractor, mode):
    """
    Extension to "extract_features" method that computes BoVW if necessary

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features

    feature_extractor: sklearn transformer
                       transformers that extract features from images

    mode: 'train' / 'validate'. Determines whether the transformers are fitted to the data or not.

    batch_size: number of images to extract features from in each process
                total memory used at a given time is cpu_cout * memory occupied
                by batch_size number of images
    """

    def contains_bovw(feature_extractor):
        if isinstance(feature_extractor, Pipeline):
            return any(isinstance(f, BoVW) for f in feature_extractor.named_steps.values())

        # TODO: identify if there is a pipeline inside feature union and decompose
        # elif isinstance(feature_extractor, FeatureUnion):
        #     for extractor in feature_extractor.transformer_list:
        #         if isinstance(extractor, Pipeline):
        #             if any(isinstance(f, BoVW) for f in feature_extractor.named_steps.values()):
        #                 return True

        return False

    if contains_bovw(feature_extractor):
        # TODO: generalize to feature union. Currently it assumes it is a pipeline
        return extract_local_features(imgs_collection, feature_extractor, mode)

    else:
        return extract_global_features(imgs_collection, feature_extractor, mode)


def complete_pipeline(feature_extractor):
    """
    1. extract features of the training and validation set, using the feature_extractor argument
    2. train model on training data
    3. make predictions on validation data
    4. evaluate performance

    Parameters
    ----------
    feature_extractor: sklearn transformer | sklearn Pipeline | sklearn FeatureUnion
                       transformers that extract features from images
    """
    # load data
    train_imgs = load_image_collection(TRAIN_PATH)
    validation_imgs = load_image_collection(VALIDATION_PATH)

    train_labels = pd.read_csv(os.path.join(DATASETS_PATH, 'train_labels.csv'))
    train_labels = extract_label_values(train_labels)

    # extract features
    t_start_features = time()
    train_features = extract_features(train_imgs, feature_extractor, mode='train')
    t_end_features = time()
    validation_features = extract_features(validation_imgs, feature_extractor, mode='validate')

    # train model
    classifier = RandomForestClassifier(n_estimators=500)
    t_start_training = time()
    classifier.fit(train_features, train_labels)
    t_end_training = time()

    # make predictions
    validation_predictions = classifier.predict(validation_features)

    # evaluate performance
    mean_f2, per_class_f2 = evaluate_performance_validation(validation_predictions)

    # print(classifier.feature_importances_)

    r = {'mean_f2': mean_f2,
         'per_class_f2': per_class_f2,
         'time_feature_extraction': t_end_features - t_start_features,
         'time_training': t_end_training - t_start_training}

    return r
