These scripts were developed to process 290 images from the Orion Nebula. They require OpenCV 3.0 and
Numpy.

clean.py holds "utility functions" to simplify the batch processing code.

batch.py is the script that effectively works and processes the images in the TIF directory
(according to the naming scheme my camera follows) and then outputs in the OUT directory.

Call is just running `python batch.py` and it works on its own. For every image, it approximates a background and subtracts it from the original. This process helps to clean light polution.
