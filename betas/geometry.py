import numpy as np

def vec2d(x, y):
    return np.array((x, y), dtype=np.float64)

def vec2di(x, y):
    return np.array((round(x), round(y)), dtype=np.int64)

def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def dot(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]

def ori(a, u, v):
    return sgn(cross(u - a, v - a))

def area2(a, b, c):
    return cross(b - a, c - b)

def ons(a, b, p):
    return not ori(a, b, p) and all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in range(2))

def segment_intersects(p, q):
    p1, p2 = ori(p[0], p[1], q[0]), ori(p[0], p[1], q[1])
    q1, q2 = ori(q[0], q[1], p[0]), ori(q[0], q[1], p[1])
    if p1 != p2 and q1 != q2:
        return True
    elif ons(p[0], p[1], q[0]) or ons(p[0], p[1], q[1]) or \
         ons(q[0], q[1], p[0]) or ons(q[0], q[1], p[1]):
        return True
    return False


class Polygon:
    @staticmethod
    def vec(x, y):
        return vec2d(x, y)

    def __init__(self, vertices):
        self.vertices = [self.vec(x, y) for x, y in vertices]

    def __repr__(self):
        return "Polygon({})".format(repr([tuple(v) for v in self.vertices]))

    def edge(self, i):
        if i not in range(len(self.vertices)):
            i %= len(self.vertices)
        if i == len(self.vertices) - 1:
            return (self.vertices[-1], self.vertices[0])
        return np.array((self.vertices[i], self.vertices[i+1]))

    @property
    def edges(self):
        return [self.edge(i) for i in range(len(self.vertices))]


class PolygonI(Polygon):
    @staticmethod
    def vec(x, y):
        return vec2di(x, y)

    def contains_point(self, point):
        pivot = vec2di(point[0] + 1, max(v[1] for v in self.vertices) + 1)
        segment = np.array((point, pivot))
        return sum(segment_intersects(segment, edge) for edge in self.edges) % 2 == 1

    def contains_segment(self, segment):
        return self.contains_point(segment[0]) and self.contains_point(segment[1]) and \
               not any(segment_intersects(segment, edge) for edge in self.edges)

    def contains_poly(self, poly):
        return all(self.contains_segment(edge) for edge in poly.edges)

    def contains(self, other):
        if is_point(other, True):
            return self.contains_point(other)
        elif is_segment(other, True):
            return self.contains_segment(other)
        elif is_polygon(other, True):
            return self.contains_poly(other)
        else:
            raise TypeError("Type mismatch for intersection check")

def polygon_intersects(a, b):
    return any(a.contains(v) for v in b.vertices) or \
           any(b.contains(v) for v in a.vertices)


# General

def is_point(a, int_=False):
    return isinstance(a, np.ndarray) and a.shape == (2,) and (not int_ or a.dtype == np.int64)

def is_segment(a, int_=False):
    return isinstance(a, np.ndarray) and a.shape == (2, 2) and (not int_ or a.dtype == np.int64)

def is_polygon(a, int_=False):
    return isinstance(a, Polygon) and (not int_ or isinstance(a, PolygonI))

def intersects(a, b):
    if is_segment(a, True) and is_segment(b, True):
        return segment_intersects(a, b)
    elif is_polygon(a, True) and is_polygon(b, True):
        return polygon_intersects(a, b)
    else:
        raise TypeError("Type mismatch for intersection check")

def contains(a, b):
    return a.contains(b)

def main():
    a = PolygonI([[0, 0], [0, 2], [2, 2], [2, 0]])
    print(list(a.vertices))
    print(list(a.edges))
    b = PolygonI(vec2di(1, 1) + v for v in a.vertices)
    print(list(b.vertices))
    print(list(b.edges))
    print(intersects(a, b))

    p = PolygonI(((34, -348), (992, -337), (994, 193), (41, 189)))
    q = PolygonI(((50, -400), (50, -350), (100, -350), (100, -400)))
    print('pq', p.contains(q))
    for v in q.vertices:
        print(v, p.contains(v))

if __name__ == '__main__':
    main()
