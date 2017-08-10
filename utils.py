import scipy.misc
import numpy as np

def get_image(image_path, image_size, nb_channels=3, is_crop=True):
    return transform(imread(image_path, nb_channels), image_size, is_crop)

def imread(path, nb_channels):
    if nb_channels == 3:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    elif nb_channels == 1:
        return scipy.misc.imread(path, mode='L').astype(np.float)

def transform(image, npx=96, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def center_crop(x, crop_h, crop_w=None, resize_w=96):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

def imsave(image, path):
    return scipy.misc.imsave(path, image)

def save_images(images, image_path):
    return imsave(inverse_transform(images), image_path)
