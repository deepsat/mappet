def infer_im_to_enu(ground, other):
    src, dst = feature_matching.find_keypoint_matches(ground.image, other.image)
    base, src1, dst1 = feature_matching.compute_homography(src, dst)
    x1 = complex(ground.image.shape[1]//2, ground.image.shape[0]//2)
    x2 = complex(*warp_perspective(base, (other.image.shape[1]//2, other.image.shape[0]//2)))
    y1 = complex(ground.metadata.x, ground.metadata.y)
    y2 = complex(other.metadata.x, other.metadata.y)
    print(f"{x1} -> {y1}, {x2} -> {y2}")
    # a*x + b = y
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    print(f"{a}*x + {b}: {y1} ~ {a*x1 + b}")
    mat = np.array([
        [a.real, -a.imag, b.real],
        [+a.imag, a.real, b.imag],
        [0, 0, 1]
    ])
    return mat


if __name__ == '__main__':
    from drone_test_data import frames_at
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'
    n = 3
    frames = frames_at([4525, 4625, 4700, 4800, 4900][:n], FILENAME, SUB_FILENAME, silent=False)
    for frame in frames:
        frame.image = cv2.resize(frame.image, (960, 540))
    frames[2].image = cv2.resize(frames[2].image[120:-120, 75:-75], (960, 540))

    def rads(i):
        return (geodesy.math.radians(x) for x in frames[i].position[:2][::-1])

    lat0, lng0 = rads(0)
    print(frames[0].position[:2][::-1], (lat0, lng0))
    plane = geodesy.LocalTangentPlane(lat0, lng0, 0)
    photos = [
        MapPhoto.from_drone_photo(DronePhoto(
            frame.image, DroneCameraMetadata(*rads(i), frames[0].altitude, 0, 0, None, None)
        ), plane, 130) for i, frame in enumerate(frames)
    ]
    print(photos)
    photos[0].metadata.yaw = 0
    imenu = infer_im_to_enu(photos[0], photos[1])
    for k in range(n):
        photos[k].im_to_enu = imenu
    for k in range(1, n):
        photos[k].homography_enhancement(photos[k-1])
    print(photos)
    for k in range(n):
        print(photos[k].transform)

        print(photos[k].transform @ ((960//2,), (540//2,), (1,)))
        cx, cy = warp_perspective(photos[k].transform, (960 // 2, 540 // 2))
        dx, dy = warp_perspective(imenu @ photos[k].transform, (960 // 2, 540 // 2))
        print((cx, cy), (photos[k].metadata.x, photos[k].metadata.y))
        print('->', (dx, dy))
        cv2.imwrite(f'test/im{k}.png', photos[k].image)
        cv2.imwrite(f'test/imw{k}.png', photos[k].warp())