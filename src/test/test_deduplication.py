from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from deduplication import LSHDeduplicator, BruteForceDeduplicator


def create_all_images_data(path):
    """
    Creates variations of the images in the test set with different brightness, translation, and rotation.
    :param path: Path to the folder containing the test images.
    :return: A list of lists, grouping together the augmented images.
    """
    path = Path(path)
    result = []
    for f in path.glob("*.jpg"):
        img = Image.open(f)
        img_bright = ImageEnhance.Brightness(img).enhance(1.5)
        img_dark = ImageEnhance.Brightness(img).enhance(0.5)
        img_shifted = img.transform(img.size, Image.AFFINE, (1, 0, 5, 0, 1, -5))
        img_rotated = img.rotate(10)
        result.append([
            np.array(img),
            np.array(img_bright),
            np.array(img_dark),
            np.array(img_shifted),
            np.array(img_rotated)])
    return np.array(result)


def create_single_image_data(path):
    """
    Creates variations of the image specified by path with different brightness, translation, and rotation
    :param path: Path to the image.
    :return: A list containing the augmented images and the original image.
    """
    path = Path(path)
    img = Image.open(path)
    img_bright = ImageEnhance.Brightness(img).enhance(1.5)
    img_dark = ImageEnhance.Brightness(img).enhance(0.5)
    img_shifted = img.transform(img.size, Image.AFFINE, (1, 0, 5, 0, 1, -5))
    img_rotated = img.rotate(10)

    return np.array([
        np.array(img),
        np.array(img_bright),
        np.array(img_dark),
        np.array(img_shifted),
        np.array(img_rotated)])


def test_LSHDeduplicator_single_image():
    """
    Tests the LSHDeduplicator on a single augmented image.
    """
    deduplicator = LSHDeduplicator()
    data = create_single_image_data("../data/2092.jpg")
    result = deduplicator.deduplicate(data)
    assert len(result) == 3


def test_LSHDeduplicator_all_images():
    """
    Tests the LSHDeduplicator on all images in the dataset.
    """
    deduplicator = LSHDeduplicator()
    data = create_all_images_data("../data").flatten()
    result = deduplicator.deduplicate(data)
    assert len(result) <= (200 * 3) + 50


def test_BruteForceDeduplicator_single_image():
    """
    Tests the BruteForceDeduplicator on a single image.
    """
    deduplicator = BruteForceDeduplicator()
    data = create_single_image_data("../data/2092.jpg")
    result = deduplicator.deduplicate(data)
    assert len(result) == 3


def test_BruteForceDeduplicator_all_images():
    """
    Tests the BruteForceDeduplicator on all images in the dataset.
    """
    deduplicator = BruteForceDeduplicator()
    data = create_all_images_data("../data").flatten()
    result = deduplicator.deduplicate(data)
    assert len(result) <= 200 * 3
