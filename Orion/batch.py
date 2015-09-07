# coding=utf-8
import clean
import cv2
import numpy as np


# utility function that helps populate the threshold value for each tuple in the list numbers.
def threshold_values(x):
    return int(round(7.0 + (x / 290.0) * 5.0))


# variables
dilate_kernel = np.array(
    [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    dtype = np.uint8
)

# this list holds two values for each element: the number of the picture and the threshold value
# that the star mask needs to be generated properly. threshold values were found through trial and
# error
numbers = [(x, threshold_values(x)) for x in xrange(1, 290 + 1)]
for x, th in numbers:
    print "\rProcessing IMG_%04d.tif..." % x,
    img = cv2.imread("TIF/IMG_%04d.TIF" % x, -1)
    height, width = img.shape[0], img.shape[1]
    stars = clean.threshold(img, th, gauss = (5,5)).astype(np.float32)
    stars = cv2.dilate(stars, dilate_kernel, iterations=1)
    stars /= 255.  # normalize the result
    stars = (stars - 1.) / -1  # we need the inverse mask
    background = clean.get_background(img, stars)
    finalimg = img - background
    # now we have to clip everything below zero:
    finalimg = np.maximum(finalimg, 0)
    #... and we stretch the levels
    finalimg /= finalimg.max()
    finalimg *= 65535
    finalimg = cv2.resize(img, (5000, 3750), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("OUT/IMG_%04d.tif" % x, finalimg.astype(np.uint16))

