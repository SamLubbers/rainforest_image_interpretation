import os
import pandas as pd
from skimage import io
from spectral import imshow, get_rgb

import sys
sys.path.append("../")
from config import TRAIN_PATH

def load_image(file_name, imgs_path=TRAIN_PATH):
    path = os.path.join(imgs_path, file_name)
    return io.imread(path)

def tif_to_rgb(img):
    """returns image with rgb values between 0 and 1"""
    # [2, 1, 0] index is needed because colours are stored as bgr
    return get_rgb(img, [2,1,0]) 

def show_tif_img(file_name, imgs_path=TRAIN_PATH):
    img = load_image(file_name, imgs_path)
    rgb = tif_to_rgb(img)
    imshow(rgb)