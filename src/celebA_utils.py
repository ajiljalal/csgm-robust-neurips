"""Some utils for celebA dataset"""

import numpy as np
import utils
import matplotlib.pyplot as plt
import png

def transpose(image):
    if image.shape[2] == 3:
        return image
    elif image.shape[0] == 3:
        return image.transpose(1,2,0)
    else:
        raise ValueError(f"image has weird shape of {image.shape}. color channel must be first or last dimension")


def scale(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val)/(max_val - min_val)

def view_image(image, hparams, mask=None):
    """Process and show the image"""
    if len(image) == hparams.n_input:
        image = image.reshape(hparams.image_shape)
        image = transpose(image)
        if mask is not None:
            mask = mask.reshape(hparams.image_shape)
            image = np.maximum(np.minimum(1.0, image - 1.0*image*(1-mask)), -1.0)
    utils.plot_image(scale(image))

def save_image(image, path):
    """Save an image as a png file"""
    x_png = np.uint8(np.clip(scale(image)*255,0,255))
    x_png = transpose(x_png)
    image_size = x_png.shape[0]
    x_png = x_png.reshape(image_size,image_size*3)
    y = png.from_array(x_png, mode='RGB')
    y.save(path)

    '''
    png_writer = png.Writer(64, 64, greyscale=False)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image.reshape([64,-1]))
    '''
