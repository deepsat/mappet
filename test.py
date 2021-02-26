import logging

import tqdm
import numpy as np
import cv2

from drone_test_data import get_drone_photos
from series import RelativeSeries
import feature_matching
import stitching

# noinspection PyArgumentList
logging.basicConfig(
    filename='test/latest.log', encoding='utf-8', level=logging.DEBUG,
    format='[%(levelname)s::%(name)s@%(relativeCreated)dms] %(message)s'
)

FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'

n = 3
phs = get_drone_photos(
    ([4525, 4625, 4700, 4800, 4850, 4900, 4950, 5000] + list(range(5000, 7000, 25)))[:n],
    FILENAME, SUB_FILENAME, silent=False
)
lat, lng, lh = phs[0].metadata.latitude, phs[0].metadata.longitude, phs[0].metadata.height

series = RelativeSeries(lat, lng, lh)
for ph in tqdm.tqdm(phs):
    series.append(ph, rel_method='even_similarity')

im_to_enu = series.fit_im_to_enu()

img_k = phs[0].image.copy()
cv2.drawKeypoints(img_k, feature_matching.get_features(img_k)[0], img_k, color=(0, 0, 255))
cv2.imwrite('test/keypoints.png', img_k)
cv2.imwrite('test/warp-ctr.png', series.warped_contour(1, 5, (0, 255, 0))[1])
img = series.stitch()
img2 = series.stitch_contours(5, (0, 255, 0))
img3 = img.copy()
stitching.lay_over(img2, 0, 0, img3)
img4 = series.stitch_contours(5, (0, 255, 0), True)
img5 = img.copy()
stitching.lay_over(img4, 0, 0, img5)
cv2.imwrite('test/stitch.png', img)
cv2.imwrite('test/stitch-ctr.png', img2)
cv2.imwrite('test/stitch-with-ctr.png', img3)
cv2.imwrite('test/stitch-ctr-full.png', img4)
cv2.imwrite('test/stitch-with-ctr-full.png', img5)
