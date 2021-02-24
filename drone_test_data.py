import datetime
import math
import typing

import tqdm
import numpy as np
import cv2

from photo import DronePhoto, DroneCameraMetadata

SUB_SEC_GAP = 1.0


class Frame:
    image: np.array
    altitude: float
    position: typing.Tuple[float, float, float]
    datetime: datetime.datetime

    def __init__(self, image: np.array, altitude: float, position: typing.Tuple[float, float, float], datetime_):
        self.image, self.altitude, self.position, self.datetime = image, altitude, position, dt

    def __repr__(self) -> str:
        return f"Frame(image=<{'x'.join(str(x) for x in self.image.shape)}>, altitude={self.altitude}, " + \
               f"position={self.position}, datetime={self.datetime})"


def parse_subtitles(sub_filename: str) -> typing.List[typing.List[str]]:
    subs = open(sub_filename, 'r').read().splitlines()
    subs = [line for line in subs if line]
    subs = [line for i, line in enumerate(subs) if i % 5 in (2, 3, 4)]
    subs = [' '.join((subs[3*i+0], subs[3*i+1], subs[3*i+2])).strip().split() for i in range(len(subs)//3)]
    return subs


def read_subtitle_data(verse: typing.List[str]) -> typing.Tuple[float, typing.Tuple[float, float, float], datetime.datetime]:
    lat, lng, th = (float(x) for x in verse[3][4:-1].split(','))
    return (
        float(verse[4].split(':')[1]),  # altitude
        (lat, lng, th),  # position
        datetime.datetime.fromisoformat(' '.join((verse[1].replace('.', '-'), verse[2]))),  # datetime
    )


def frames_at(indices : typing.List[int], filename: str, sub_filename: typing.Optional[str] = None,
              silent: bool = True) -> typing.List[Frame]:
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not silent:
        print(f"Video at fps={fps}")
    if sub_filename is not None:
        subs = parse_subtitles(sub_filename)
    else:
        subs = None
    indices = sorted(indices)
    result = []
    if not silent:
        indices = tqdm.tqdm(indices)
    for i in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = video.read()
        assert success
        if subs:
            s = min(max(int(i // (SUB_SEC_GAP * fps)) - 1, 0), len(subs) - 1)
            altitude, position, dt = read_subtitle_data(subs[s])
        else:
            altitude, position, dt = None, None, None
        frame = Frame(image, altitude, position, dt)
        result.append(frame)
    return result


def get_drone_photos(indices : typing.List[int], filename: str, sub_filename: typing.Optional[str] = None,
                     silent: bool = True) -> typing.List[DronePhoto]:
    frames = frames_at(indices, filename, sub_filename, silent)
    return [DronePhoto(
        frame.image, DroneCameraMetadata(
            math.radians(frame.position[1]), math.radians(frame.position[0]), frame.altitude,
            0, 0, None, frame.datetime, None
        )
    ) for frame in frames]
