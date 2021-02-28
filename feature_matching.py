import logging
import functools
import typing

import numpy as np
import cv2

log = logging.getLogger('mappet.feature_matching')

IMAGE_FEATURE_CACHE = 4
default_key_detect = cv2.ORB_create(nfeatures=2**12)
default_desc_detect = cv2.SIFT_create()
default_matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 4}, {'checks': 64})


@functools.lru_cache(maxsize=IMAGE_FEATURE_CACHE)
def _get_features(data: bytes, shape: typing.Tuple[int, int], key_detect, desc_detect):
    image = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    key = key_detect.detect(image, None)
    key, desc = desc_detect.compute(image, key)
    return key, desc


def get_features(image: np.array, *, key_detect=default_key_detect, desc_detect=default_desc_detect):
    """
    Caches up to `IMAGE_FEATURE_CACHE` results in LRU fashion.
    :param image: Image to extract features from.
    :param key_detect: Used keypoint detection method (cv2 object, default is ORB).
    :param desc_detect: Used descriptor extraction method (cv2 object, default is SIFT).
    :return: Keypoints and descriptors in default cv2 format.
    """
    return _get_features(image.data.tobytes(), image.shape, key_detect, desc_detect)


def find_keypoint_matches(first: np.array, second: np.array, *, lowe_coefficient: float = 0.8, matcher=default_matcher):
    """
    :param first: Image to match against (destination)
    :param second: Image to match from (source)
    :param lowe_coefficient: Coefficient for Lowe's ratio test. Should be 0.6-0.8, higher means less refined matches.
    :param matcher: Used matching method (cv2 object, default is FLANN).
    :return:
    """
    first_key,  first_desc = get_features(first)
    second_key, second_desc = get_features(second)
    matches = matcher.knnMatch(second_desc, first_desc, k=2)
    matches = [m for m, n in matches if m.distance < lowe_coefficient * n.distance]
    src = np.array([second_key[match.queryIdx].pt for match in matches]).reshape((-1, 2))
    dst = np.array([first_key[match.trainIdx].pt for match in matches]).reshape((-1, 2))
    log.debug(f"find_keypoint_matches: ({len(first_key)}, {len(second_key)}) keypoints yielded {len(matches)} matches")
    return src, dst


