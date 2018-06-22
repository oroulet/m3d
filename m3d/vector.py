import numpy as np

from m3d.common import float_eps


class Vector(object):
    """
    Represent a vector.
    Takes either x, y, z as argument or a list/array
    """

    def __init__(self, x=0.0, y=0.0, z=0.0, dtype=np.float32):
        if isinstance(x, (list, tuple)):
            self._data = np.array(x, dtype=dtype)
        elif isinstance(x, np.ndarray):
            self._data = x
        else:
            self._data = np.array([float(x), float(y), float(z)], dtype=dtype)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        self._data[idx] = val

    @property
    def x(self):
        return float(self._data[0])

    @x.setter
    def x(self, val):
        self._data[0] = val

    @property
    def y(self):
        return float(self._data[1])

    @y.setter
    def y(self, val):
        self._data[1] = val

    @property
    def z(self):
        return float(self._data[2])

    @z.setter
    def z(self, val):
        self._data[2] = val

    def __str__(self):
        return "Vector({}, {}, {})".format(self.x, self.y, self.z)

    def __sub__(self, other):
        return Vector(self.data - other.data)

    def __add__(self, other):
        return Vector(self.data + other.data)

    __repr__ = __str__

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return (abs(self.data - other.data) <= float_eps).all()

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(self._data * other)
        else:
            raise ValueError()

    @property
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def dist(self, other):
        v = Vector(other.data - self.data)
        return v.length
