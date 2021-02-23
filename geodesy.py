import math
import typing

import numpy as np

# Semi-major axes from https://en.wikipedia.org/wiki/Earth_radius
A = 6378137.0
B = 6356752.3
# Formulas based on https://en.wikipedia.org/wiki/Geographic_coordinate_conversion


def geodetic_to_ecef(latitude: float, longitude: float, height: float) -> typing.Tuple[float, float, float]:
    r = A**2 / math.sqrt((A * math.cos(latitude))**2 + (B * math.sin(latitude))**2)
    t = (r + height) * math.cos(latitude)
    return t * math.cos(longitude), t * math.sin(longitude), (B**2/A**2 * r + height) * math.sin(latitude)


class LocalTangentPlane:
    def __init__(self, latitude, longitude, height):
        self.latitude, self.longitude, self.height = latitude, longitude, height

    def enu(self, latitude, longitude, height):
        x0, y0, z0 = geodetic_to_ecef(self.latitude, self.longitude, self.height)
        x, y, z = geodetic_to_ecef(latitude, longitude, height)
        x -= x0
        y -= y0
        z -= z0
        lng, lat = self.longitude, self.latitude
        return (
            round(-math.sin(lng) * x + math.cos(lng) * y, 3),
            round(-math.sin(lat) * math.cos(lng) * x + -math.sin(lat) * math.sin(lng) * y + math.cos(lat) * z, 3),
            round(math.cos(lat) * math.cos(lng) * x + math.cos(lat) * math.sin(lng) * y + math.sin(lat) * z, 3)
        )


if __name__ == '__main__':
    rad = math.radians
    plane = LocalTangentPlane(rad(52.284509), rad(20.804014), 10)
    print(plane.enu(rad(52.284509), rad(20.804014), 10))
    print(plane.enu(rad(52.284509), rad(20.804014), 50))
    print(plane.enu(rad(52.285618), rad(20.804297), 10))
    print(plane.enu(rad(52.285194), rad(20.800139), 10))