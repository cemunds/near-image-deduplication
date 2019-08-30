from hashes import dhash
from PIL import Image
import numpy as np
from pathlib import Path


def test_dhash():
    """
    Tests whether every image in the test dataset can be hashed into a 64-bit hash.
    """
    path = Path("../data")

    for f in path.glob("*.jpg"):
        img = np.array(Image.open(f))
        h = dhash(img)
        assert len(h) == 64
