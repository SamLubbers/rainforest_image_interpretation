import os
import sys
from time import process_time

import numpy as np
import pandas as pd
from skimage.io import concatenate_images
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../")
from config import DATASETS_PATH, TRAIN_PATH, VALIDATION_PATH
from helpers.img_utils import load_image_collection
from helpers.data import extract_label_values
from helpers.evaluation import evaluate_performance_validation


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
    t_start_features = process_time()
    train_features = extract_features(train_imgs, feature_extractor)
    t_end_features = process_time()
    validation_features = extract_features(validation_imgs, feature_extractor)

    # train model
    classifier = RandomForestClassifier(n_estimators=500)
    t_start_training = process_time()
    classifier.fit(train_features, train_labels)
    t_end_training = process_time()

    # make predictions
    validation_predictions = classifier.predict(validation_features)

    # evaluate performance
    mean_f2, per_class_f2 = evaluate_performance_validation(validation_predictions)

    r = {'mean_f2': mean_f2,
         'per_class_f2': per_class_f2,
         'time_feature_extraction': t_end_features - t_start_features,
         'time_training': t_end_training - t_start_training}

    return r
