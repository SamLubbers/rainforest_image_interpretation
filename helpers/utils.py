import os
import pandas as pd
from skimage.io import imread, imshow, imread_collection_wrapper
from spectral import get_rgb

import sys
sys.path.append("../")
from config import TRAIN_PATH

def load_image(file_name, imgs_path=TRAIN_PATH):
    path = os.path.join(imgs_path, file_name)
    return imread(path)

def load_image_collection(imgs_path):
    img_file_pattern = "*.tif"
    imgs_path_pattern = os.path.join(imgs_path, img_file_pattern)
    """
    it is necessary to create custom imread collection function which reads images with the 'imread' function
    in order to obtain the raw values from the tif image. 
    The default imread_collection function returns images that are uncorrectly scaled between 0 and 255
    """
    imread_collection_custom = imread_collection_wrapper(imread)
    return imread_collection_custom(imgs_path_pattern, conserve_memory=True)

def tif_to_rgb(img):
    """returns image with rgb values between 0 and 1"""
    # [2, 1, 0] index is needed because colours are stored as bgr (wavelength order)
    return get_rgb(img, [2,1,0]) 

def load_and_show_img(file_name, imgs_path=TRAIN_PATH):
    img = load_image(file_name, imgs_path)
    rgb = tif_to_rgb(img)
    imshow(rgb)
    
def extract_label_values(df, values_col=2):
    """
    extract values from dataframe of labels
    
    Parameters
    ----------
    df : pandas.DataFrame
         labels
    
    values_col: int
                column index of df where values start 
    """
    return df.iloc[:, 2:].values
