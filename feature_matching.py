import numpy as np
import cv2

default_key_detect = cv2.ORB_create(nfeatures=2**14)
default_desc_detect = cv2.SIFT_create()
default_matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 4}, {'checks': 64})


def get_features(image, *, key_detect=default_key_detect, desc_detect=default_desc_detect):
    key = key_detect.detect(image, None)
    key, desc = desc_detect.compute(image, key)
    return key, desc


def find_keypoint_matches(first: np.array, second: np.array, *, lowe_coefficient: float = 0.8, matcher=default_matcher):
    first_key,  first_desc = get_features(first)
    second_key, second_desc = get_features(second)
    matches = matcher.knnMatch(second_desc, first_desc, k=2)
    matches = [m for m, n in matches if m.distance < lowe_coefficient * n.distance]
    src = np.array([second_key[match.queryIdx].pt for match in matches]).reshape((-1, 1, 2))
    dst = np.array([first_key[match.trainIdx].pt for match in matches]).reshape((-1, 1, 2))
    return src, dst


def compute_homography(src, dst):
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask = mask.reshape(-1).astype(bool)
    print(f"{100 * mask.sum() / mask.size:.1f}% inliers")
    src, dst = src[mask], dst[mask]
    base = homography.copy()
    return base, src, dst


def compute_even_similarity(src0, dst0):
    src = np.array([complex(pt[0], pt[1]) for pt in src0])
    dst = np.array([complex(pt[0], pt[1]) for pt in dst0])
    raise NotImplementedError
