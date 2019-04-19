import os
import sys
from time import time

import numpy as np
import pandas as pd
from skimage.io import concatenate_images
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../")
from config import DATASETS_PATH, TRAIN_PATH, VALIDATION_PATH
from helpers.img_utils import load_image_collection
from helpers.data import extract_label_values
from helpers.evaluation import evaluate_performance_validation
from multiprocessing import Pool, cpu_count


def extract_features_batch(imgs_collection_batch, feature_extractor):
    """extracts features of a collection of images.
    It is required as a separate function for parallel processing"""

    imgs_batch = concatenate_images(imgs_collection_batch)
    return feature_extractor.fit_transform(imgs_batch)

def extract_features(imgs_collection, feature_extractor, batch_size=200):
    """
    Extracts from imgs_collection the set of features specified in feature_extractor.
    Extracts the features in batches because it is unviable to load all images at once into memory

    Parameters
    ----------
    imgs_collection : skimage.ImageCollection
                      collection of images from which we want to extract the features
    feature_extractor: sklearn transformer
                       transformers that extract features from images
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
        batch_features = [pool.apply(extract_features_batch, args=(ims, feature_extractor))
                          for ims in imgs_collection_batches]

    features = np.concatenate(batch_features)
    return features


def complete_pipeline(feature_extractor):
    """
    1. extract features of the training and validation set, using the feature_extractor argument
    2. train model on training data
    3. make predictions on validation data
    4. evaluate performance

    Parameters
    ----------
    feature_extractor: sklearn transformer
                       transformers that extract features from images
    """
    # load data
    train_imgs = load_image_collection(TRAIN_PATH)
    validation_imgs = load_image_collection(VALIDATION_PATH)

    train_labels = pd.read_csv(os.path.join(DATASETS_PATH, 'train_labels.csv'))
    train_labels = extract_label_values(train_labels)

    # extract features
    t_start_features = time()
    train_features = extract_features(train_imgs, feature_extractor)
    t_end_features = time()
    validation_features = extract_features(validation_imgs, feature_extractor)

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
