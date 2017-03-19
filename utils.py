import os
import numpy as np
import scipy.misc
from glob import glob

def input_data(data_path,
    input_height, input_width, resize_height=64, resize_width=64,
    is_grayscale=False):
    data_files = glob(os.path.join(data_path, '*.jpg'))
    input_images = [transform(get_image(data_file, is_grayscale),
        input_height, input_width, resize_height, resize_width)
        for data_file in data_files]
    if(is_grayscale):
        return np.array(input_images).astype(np.float32)[:, :, :, None]
    else:
        return np.array(input_images).astype(np.float32)

def get_image(data_path, is_grayscale=False):
    image = imread(data_path, is_grayscale)
    return image

def imread(data_path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(data_path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(data_path).astype(np.float)

def transform(image,
    input_height, input_width, resize_height=64, resize_width=64):
    new_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(new_image)/127.5 - 1.
