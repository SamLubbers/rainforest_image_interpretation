import os
import sys
from time import time

import numpy as np
import pandas as pd
from skimage.io import concatenate_images
from sklearn.ensemble import RandomForestClassifier
from itertools import chain
from sklearn.pipeline import make_union, Pipeline, FeatureUnion

from feature_extractors import BoVW

sys.path.append("../")
from config import DATASETS_PATH, TRAIN_PATH, VALIDATION_PATH
from helpers.img_utils import load_image_collection
from helpers.data import extract_label_values
from helpers.evaluation import evaluate_performance_validation
from joblib import Parallel, delayed, parallel_backend


def extract_features_batch(imgs_collection_batch, feature_extractor, mode):
    """extracts features of a collection of images
    requires as separate function for parallel processing
    """

    imgs_batch = concatenate_images(imgs_collection_batch)
    if mode == 'train':
        feature_extractor.fit(imgs_batch)

    return feature_extractor.transform(imgs_batch)


def extract_features(imgs_collection, feature_extractor, mode, batch_size=200, feature_type='global'):
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
                total memory used at a given time is n_jobs * (memory occupied
                by batch_size number of images)

    feature_type: 'global' / 'local'

    Returns
    -------
    features: numpy.ndarray (np.float64)
              set of features extracted from the image,
              with one row for each image and one column for each feature
    """

    n_images = len(imgs_collection)

    imgs_collection_batches = [imgs_collection[i:i + batch_size]
                               for i in range(0, n_images, batch_size)]

    if feature_type == 'global':
        pll_backend = 'loky'
        n_jobs = -1  # use all processors
    else:
        pll_backend = 'threading'  # loky multiprocessing backend does not work for cv2 features
        n_jobs = 8  # use 8 threads

    with parallel_backend(pll_backend, n_jobs=n_jobs):
        batch_features = Parallel()(delayed(extract_features_batch)
                                    (imgs, feature_extractor, mode)
                                    for imgs in imgs_collection_batches)

    if feature_type == 'global':
        return np.concatenate(batch_features)
    elif feature_type == 'local':
        return list(chain.from_iterable(batch_features))


def extract_local_bovw(imgs_collection, local_feature_pipeline, mode):
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
    local_feature_descriptors = extract_features(imgs_collection,
                                                 local_feature_extractor,
                                                 mode,
                                                 feature_type='local')

    if mode == 'train':
        # create dictionary
        bovw.fit(local_feature_descriptors)

    # convert local features to bovw representation
    return bovw.transform(local_feature_descriptors)


def extract_all_features(imgs_collection, feature_extractor, mode):
    """
    feature extraction of local and global features

    This function is necessary because it is not possible to extract local features directly
    with `extract_features`. Instead, if any local features are to be extracted this method calls
    the specialised `extract_local_features` method.

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features

    feature_extractor: sklearn transformer | sklearn Pipeline | sklearn FeatureUnion
                       transformers that extract features from images
                       It can also contain a bovw extractor

    mode: 'train' / 'validate'. Determines whether the transformers are fitted to the data or not.

    batch_size: number of images to extract features from in each process
                total memory used at a given time is cpu_cout * memory occupied
                by batch_size number of images
    """

    def is_local_pipeline(feature_extractor):
        if isinstance(feature_extractor, Pipeline):
            return any(isinstance(f, BoVW) for f in feature_extractor.named_steps.values())
        else:
            return False

    if is_local_pipeline(feature_extractor):
        return extract_local_bovw(imgs_collection, feature_extractor, mode)

    elif isinstance(feature_extractor, FeatureUnion):
        extractors = [x for _, x in feature_extractor.transformer_list]
        if any([is_local_pipeline(x) for x in extractors]):
            global_extractors = []
            local_extractors = []

            for x in extractors:
                if is_local_pipeline(x):
                    local_extractors.append(x)
                else:
                    global_extractors.append(x)

            global_features = extract_features(imgs_collection,
                                               make_union(*global_extractors), mode)

            local_features = []
            for x in local_extractors:
                local_features.append(extract_local_bovw(imgs_collection, x, mode))

            local_features = np.concatenate(local_features, axis=1)
            return np.concatenate([global_features, local_features], axis=1)

    return extract_features(imgs_collection, feature_extractor, mode)


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
    train_features = extract_all_features(train_imgs, feature_extractor, mode='train')
    t_end_features = time()
    validation_features = extract_all_features(validation_imgs, feature_extractor, mode='validate')

    # train model
    classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
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
