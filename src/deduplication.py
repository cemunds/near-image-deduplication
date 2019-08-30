import numpy as np
from collections import namedtuple
from scipy.spatial.distance import hamming
from hashes import dhash


"""
Simple image container for a hashed image. Contains the image data, the calculated hash, and the found duplicates.
"""
HashedImage = namedtuple("HashedImage", ["data", "hash", "duplicates"])


class Deduplicator:
    """
    Base class for a Deduplicator.
    """

    def __init__(self, hash_func=dhash):
        """
        Constructs a Deduplicator with the given hash function. The hash function should take an image
        in numpy array format and return a numpy array with the individual bits of the hash.

        :param hash_func: The hash function to be used for fingerprinting the images.
        """
        self._hash_func = hash_func

    def _flatten(self, img):
        result = []
        result.append(img.data)
        for child in img.duplicates:
            result += self._flatten(child)
        return result

    def deduplicate(self, imgs):
        pass


class LSHDeduplicator(Deduplicator):
    """
    A Deduplicator that uses Locality-Sensitive Hashing to identify potential duplicate images. The LSH algorithm
    is orders of magnitudes faster than the brute force approach. However, it only returns an approximate
    result and might not find certain duplicates.
    """

    def __init__(self, hash_func=dhash, k=32, l=50, d=64):
        """
        Constructs an LSHDeduplicator.
        :param hash_func: The hash function to be used for fingerprinting the images.
        :param k:
        :param l: The number of hash functions to use in the LSH algorithm.
        :param d: The dimensionality of the image hash.
        """
        Deduplicator.__init__(self, hash_func)
        self._k = k
        self._l = l
        self._d = d
        self._projections = np.array([np.random.choice(list(range(d)), k) for _ in range(l)])

    def deduplicate(self, imgs):
        """
        Deduplicates the given images.
        :param imgs: Images to deduplicate.
        :return: A list of lists. Detected duplicate images are grouped together in one list and there
        is one list per detected group.
        """
        hash_table = {}
        result = []
        duplicate_count = 0

        for img in imgs:
            hashed_img = HashedImage(img, self._hash_func(img), [])
            unique = True
            for g in self._projections:
                g_x = "".join(hashed_img.hash[g].astype("str"))
                l = hash_table.get(g_x, [])

                for potential_duplicate in l:
                    if not unique:
                        break

                    if hashed_img is potential_duplicate:
                        continue

                    distance = hamming(hashed_img.hash, potential_duplicate.hash) * self._d
                    if distance < 10:
                        unique = False
                        potential_duplicate.duplicates.append(hashed_img)
                        duplicate_count += 1

                l.append(hashed_img)
                hash_table[g_x] = l

            if unique:
                result.append(hashed_img)

        result = [self._flatten(img) for img in result]
        return result


class BruteForceDeduplicator(Deduplicator):
    """
    A Deduplicator that uses a brute force approach to deduplicate images. It hashes every image
    and then performs a breadth-first-search to identify groups of duplicates, comparing every
    image to every other image that has not been put in a group, yet.
    """

    def deduplicate(self, imgs):
        """
        Deduplicates the given images.
        :param imgs: Images to deduplicate.
        :return: A list of lists. Detected duplicate images are grouped together in one list and there
        is one list per detected group.
        """
        result = []
        duplicate_count = 0
        hashed_images = []
        matched = np.array([False] * len(imgs))
        queue = []

        for img in imgs:
            hashed_img = HashedImage(img, self._hash_func(img), [])
            hashed_images.append(hashed_img)

        while not matched.all():
            unmatched = np.where(matched == False)[0]
            starting_point = unmatched[0]
            queue.append(hashed_images[starting_point])
            result.append(hashed_images[starting_point])
            matched[starting_point] = True

            while queue:
                img = queue.pop(0)
                for i in range(len(hashed_images)):
                    if img is hashed_images[i] or matched[i]:
                        continue

                    distance = hamming(img.hash, hashed_images[i].hash) * 64
                    if distance < 10:
                        img.duplicates.append(hashed_images[i])
                        queue.append(hashed_images[i])
                        matched[i] = True
                        duplicate_count += 1

        result = [self._flatten(img) for img in result]
        return result
