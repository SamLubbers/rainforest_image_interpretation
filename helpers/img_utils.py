import os
import sys

from skimage.color import rgb2gray
from skimage.io import imread, imshow, imread_collection_wrapper
from spectral import get_rgb

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

def tif_to_rgb(img, as_int=True):
    #  [2, 1, 0] index is needed because colours are stored as bgr (wavelength order)
    img_rgb = get_rgb(img, [2, 1, 0])
    # scale between 0 and 255 and convert to type uint8
    if as_int:
        img_rgb = (img_rgb * 255).astype('uint8')
    return img_rgb

def tif_to_grayscale(img, as_int=True):
    img_grayscale = rgb2gray(tif_to_rgb(img, as_int=False))
    if as_int:
        img_grayscale = (img_grayscale * 255).astype('uint8')
    return img_grayscale

def load_and_show_img(file_name, imgs_path=TRAIN_PATH):
    img = load_image(file_name, imgs_path)
    rgb = tif_to_rgb(img)
    imshow(rgb)
