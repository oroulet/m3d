import numpy as np


class Quaternion:
    """
    Represent a quaternion
    Takes a list/array of length 4 as argument.
    default to [0, 0, 0 ,0]
    """
    def __init__(self, x=None):
        if x is None:
            x = [0, 0, 0, 0]
        if isinstance(x, (list, tuple)):
            if len(x) == 4:
                self._data = np.array(x)
            else:
                raise ValueError(f"A list of length 4 is expected, got {x}")
        elif isinstance(x, np.ndarray):
            if x.shape == (4, ):
                self._data = x
            else:
                raise ValueError(f"A array of shape (4,) is expected got {x}")
        else:
            raise ValueError(f"A array of shape (4,) is expected got {x}")

    def __str__(self):
        return "Quaternion({})".format(self._data)

    __repr__ = __str__

    @property
    def data(self):
        """
        Access the numpy array used by Quaternion
        """
        return self._data

    @property
    def scalar(self):
        return self._data[0]

    @property
    def vec(self):
        return self._data[1:]

    def __rmul__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError()
        return Quaternion(self._data * other)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            s1 = self._data[0]
            s2 = other._data[0]
            q1 = self._data[1:]
            q2 = other._data[1:]
            scalar_part = np.asarray([s1 * s2 - np.dot(q1, q2)])
            imag_part = np.asarray(s1 * q2 + s2 * q1 + np.cross(q1, q2))
            return np.hstack((scalar_part, imag_part))
        if isinstance(other, (int, float)):
            return Quaternion(other * self._data)

        raise ValueError("Cannot multiply quaternion with type {}".format(type(other)))

    @property
    def conjugate(self):
        return Quaternion([self._data[0], -self._data[1], -self.data[2], -self.data[3]])

    def copy(self):
        return Quaternion(self._data.copy())
