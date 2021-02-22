import numpy as np
import cv2

default_key_detect = cv2.ORB_create(nfeatures=2**14)
default_desc_detect = cv2.SIFT_create()
default_matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 4}, {'checks': 64})


def get_features(image, *, key_detect=default_key_detect, desc_detect=default_desc_detect):
    key = key_detect.detect(image, None)
    key, desc = desc_detect.compute(image, key)
    return key, desc


def find_keypoint_matches(first, second, *, lowe_coefficient=0.8, matcher=default_matcher):
    matches = matcher.knnMatch(second.desc, first.desc, k=2)
    matches = [m for m, n in matches if m.distance < lowe_coefficient * n.distance]
    src = np.array([second.key[match.queryIdx].pt for match in matches]).reshape((-1, 1, 2))
    dst = np.array([first .key[match.trainIdx].pt for match in matches]).reshape((-1, 1, 2))
    return src, dst


def compute_homography(src, dst):
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask = mask.reshape(-1).astype(bool)
    src, dst = src[mask], dst[mask]
    return homography, src, dst


# Translate, Scale, Rotate
def compute_tsr_transformation(src, dst):
    # TODO
    raise NotImplementedError
