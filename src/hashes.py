import cv2
import numpy as np


def dhash(image, hash_size=8):
    """
    Hashes an image with the difference hash algorithm. It converts the image to grayscale and compares the
    pixel intensities of the image to another version of the same image, shifted one pixel to the right. This
    creates a 2D grid of bits, indicating at which positions the right pixel is brighter than the pixel to the
    left of it. This creates a binary hash that is invariant to image brightness and partly invariant to image scale.
    :param image: The image to hash.
    :param hash_size: The desired hash size.
    :return: A numpy array containing {hash_size} bits.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return np.array([1 if i else 0 for i in diff.flatten()])
