import numpy as np

from m3d.common import float_eps


class Vector(object):
    """
    Represent a vector.
    Takes either x, y, z as argument or a list/array
    """

    def __init__(self, x=0.0, y=0.0, z=0.0, dtype=np.float32):
        if isinstance(x, (list, tuple)):
            if len(x) == 3:
                self._data = np.array(x, dtype=dtype)
            else:
                raise ValueError(f"A list of length 3 is expected, got {x}")
        elif isinstance(x, np.ndarray):
            if x.shape == (3,):
                self._data = x
            else:
                raise ValueError(f"A array of shape (3,) is expected got {x}")
        else:
            self._data = np.array([float(x), float(y), float(z)], dtype=dtype)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        self._data[idx] = val

    def copy(self):
        return Vector(self._data.copy())

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

    def __neg__(self):
        return Vector(-self._data)

    __repr__ = __str__

    @property
    def data(self) -> np.ndarray:
        return self._data
    array = data

    def __eq__(self, other):
        return self.similar(other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(self._data * other)
        else:
            raise ValueError()

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Vector(self._data / other)
        else:
            raise ValueError()

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(self._data * other)
        else:
            raise ValueError()

    @property
    def length(self):
        return float(np.linalg.norm(self.data))

    def dist(self, other):
        """
        return abolute distance to another vector
        """
        v = Vector(other.data - self.data)
        return v.length

    def similar(self, other, tol=float_eps):
        """
        Return True if distance to other Vector is less than tol
        return False otherwise
        """
        if not isinstance(other, Vector):
            raise ValueError("Expecting a Vector object, received {} of type {}".format(other, type(other)))
        return self.dist(other) <= tol

    def normalize(self):
        """
        Normalize in place vector
        """
        if self.length == 0:
            return
        self._data /= self.length

    def normalized(self):
        """
        Return a normalized copy of vector
        """
        if self.length == 0:
            return Vector(self.data)
        return Vector(self._data / self.length)

    def cross(self, other):
        if not isinstance(other, Vector):
            other = Vector(other)
        return Vector(np.cross(self.data, other.data))

    def dot(self, other):
        if not isinstance(other, Vector):
            other = Vector(other)
        return np.dot(self.data, other.data)

    __matmul__ = dot

    def project(self, other):
        if not isinstance(other, Vector):
            other = Vector(other)
        other = other.normalized()
        return self.dot(other) * other

    def angle(self, other, normal_vector=None):
        """
        If provided, normal_vector is a vector defining the reference plane to be used to compute sign of angle.
        Otherwise, returned angle is between 0 and pi.
        """
        cos = self.dot(other) / (self.length * other.length)
        angle = np.arccos(np.clip(cos, -1, 1))
        if normal_vector is not None:
            angle = angle * np.sign(normal_vector.dot(self.cross(other)))
        return angle


# some units vectors
e0 = ex = Vector(1, 0, 0)
e1 = ey = Vector(0, 1, 0)
e2 = ez = Vector(0, 0, 1)
