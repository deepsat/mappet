import logging

import tqdm
import numpy as np
import cv2
import tensorflow as tf
import keras_segmentation

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

print("Segmentation")

segs = []
model = tf.keras.models.load_model('test/segmentation')
print(model.input_shape, '->', model.output_shape)
seg_colors = [
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
]

for p, photo in enumerate(series.photos):
    h, w = photo.image.shape[:2]
    mx, my = (w % 256) // 256, (h % 256) // 256
    tiles = []
    remap = []
    for i in range(h // 256):
        for j in range(w // 256):
            remap.append((i, j))
            tiles.append(photo.image[my+i*256:my+(i+1)*256, mx+j*256:mx+(j+1)*256])
    print(np.array(tiles).shape)
    pred = model.predict(np.array(tiles))
    print('->', pred.shape, np.argmax(pred, axis=-1).shape)
    seg_tiles2 = np.argmax(pred, axis=-1).reshape((len(tiles), 128, 128))
    seg_tiles = np.repeat(np.repeat(seg_tiles2, 2, axis=1), 2, axis=2)
    print('-->', seg_tiles.shape)
    segs.append(np.zeros((h, w, 3), dtype=np.uint8))
    for k, tile in enumerate(seg_tiles):
        i, j = remap[k]
        color_tile = np.zeros((*tile.shape, 3), np.uint8)
        for c, color in enumerate(seg_colors):
            color_tile[tile == c] = color
        segs[p][my + i * 256:my + (i + 1) * 256, mx + j * 256:mx + (j + 1) * 256] = color_tile
        if k == 0:
            cv2.imwrite(f'test/segs/tile{p}.png', tiles[k])
            cv2.imwrite(f'test/segs/seg-tile{p}.png', color_tile)

    cv2.imwrite(f'test/segs/seg{p}.png', segs[p])

# cv2.imwrite('test/stitch-seg.png', stitching.lay_on_canvas(series.bounds(), (series.warped(i, segs[i]) for i in range(len(series.photos))))

print("Showcase")
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

print("Covers")
covers, get_square = series.tile_covering()

x_min, x_max, y_min, y_max = series.bounds()
for k in range(len(series.photos)):
    image = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 3), np.uint8)
    cv2.fillPoly(image, (series.quad(k) - [x_min, y_min]).astype(np.int32).reshape((1, -1, 2)), (255, 123, 123))
    for i, j in covers[k]:
        cv2.fillPoly(image, (get_square(i, j) - [x_min, y_min]).astype(np.int32).reshape((1, -1, 2)), (255, 255, 255))
    cv2.imwrite(f'test/ghosts/{k}.png', image)

