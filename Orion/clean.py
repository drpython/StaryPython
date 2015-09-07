# coding=utf-8
import cv2
import numpy as np


# Basic functions
def get_background(img, mask, scale_factor=10, gauss=False, gauss_sigma=0):
    """
    Automatically generates a background based on the image and the stars (found through the
    threshold function). The algorithm is self-developed. Though mostly robust for star fields, it
    can take away details from very extensive nebulosities. For such images, best is to make a
    composite mask, but keep in mind that extensive masks will not work so well.
    
    @param img: Numpy array/OpenCV image.
    @param mask: Numpy arrat that us ised to mask out the stars. Must be normalized to have every
        component in the 0.0..1.0 range. mask.dtype should be float of any precision.
    @param gauss: tuple (int, int) to feed to cv2.GaussiabBlur as kernel.
    @param scale_factor: by how much the image will be downscaled before applying the gaussian
        kernel.
    """
    # set up some useful variables
    height = img.shape[0]
    width = img.shape[1]
    height_ds = height / scale_factor
    width_ds = width / scale_factor
    # since gaussian filtering is integral to this algorithm, if we are not given a kernal we
    # must make one
    if not(gauss):
        ksize = max(img.shape) / 125  # totally made up heuristic
        if ksize % 2 == 0:
            ksize -= 1
        gauss = (ksize, ksize)
    # safety check for the lazy:
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask /= mask.max()
    if mask.min() < 0.0:
        mask = np.maximum(mask, 0)
    # here begins the magic
    mask_inv = (mask - 1.0) * -1
    background = img * mask
    # we're gonna fill the dark areas of the mask with the average of the 
    uniform_colour = cv2.resize(background, (1,1), interpolation=cv2.INTER_AREA)
    uniform_colour = cv2.resize(uniform_colour, (width, height), interpolation=cv2.INTER_LINEAR)
    uniform_colour *= mask_inv
    preuc = background.copy()
    background += uniform_colour
    background = cv2.resize(background, (width_ds, height_ds), interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(background, gauss, gauss_sigma)
    background = cv2.resize(background, (width, height), interpolation=cv2.INTER_CUBIC)
    return background



def invert(img):
    """
    Inverts an image based on its dtype.
    
    @param img: NumPy array/OpenCV image.
    """
    maxval = 255.0
    if img.dtype == np.uint16:
        maxval = 65535.0
    otype = img.dtype
    img = img.astype(np.float32) / maxval
    img -= 1
    img *= -1
    img *= maxval
    return img.astype(otype)


def threshold(img, level, thtype=cv2.THRESH_BINARY, gauss=False):
    """
    Simplifies getting a clean threshold (cv2.threshold gives images that are binary, but for every
    component, so a pixel can be (255,0,0) after a threshold -- that is, the result isn't B&W). This
    function returns B&W binary images.
    
    @param img: NumPy array/OpenCV image.
    @param level: amount of threshold. 0..255 int.
    @param thtype: type of threshold operation to use. cv2 flag.
    @param gauss: tuple (int, int) to feed to cv2.GaussianBlur as kernel.
    """
    if gauss:
        img = cv2.GaussianBlur(img, gauss, 0)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    ret, th = cv2.threshold(img, level, 255, thtype)
    r, g, b = th[:,:,2], th[:,:,1], th[:,:,0]
    th = r + g + b
    th = np.minimum(th, 255)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

