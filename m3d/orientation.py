import math

import numpy as np

from m3d.vector import Vector
from m3d.common import float_eps


class Orientation(object):
    def __init__(self, data: np.ndarray = None, dtype=np.float32):
        if isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, list):
            self._data = np.array(data)
        elif data is None:
            self._data = np.identity(3, dtype=dtype)
        else:
            raise ValueError()

    def rotate_xb(self, val: float):
        t = np.array([[1, 0, 0], [0, np.cos(val), -np.sin(val)], [0, np.sin(val), np.cos(val)]])
        self._data[:] = t @ self._data

    def rotate_yb(self, val: float):
        t = np.array([[np.cos(val), 0, np.sin(val)], [0, 1, 0], [-np.sin(val), 0, np.cos(val)]])
        self._data[:] = t @ self._data

    def rotate_zb(self, val: float):
        t = np.array([[np.cos(val), -np.sin(val), 0], [np.sin(val), np.cos(val), 0], [0, 0, 1]])
        self._data[:] = t @ self._data

    def rotate_xt(self, val: float):
        t = np.array([[1, 0, 0], [0, np.cos(val), -np.sin(val)], [0, np.sin(val), np.cos(val)]])
        self._data[:] = self._data @ t

    def rotate_yt(self, val: float):
        t = np.array([[np.cos(val), 0, np.sin(val)], [0, 1, 0], [-np.sin(val), 0, np.cos(val)]])
        self._data[:] = self._data @ t

    def rotate_zt(self, val: float):
        t = np.array([[np.cos(val), -np.sin(val), 0], [np.sin(val), np.cos(val), 0], [0, 0, 1]])
        self._data[:] = self._data @ t

    def __str__(self):
        return "Orientation(\n{}\n)".format(self.data)

    __repr__ = __str__

    def inverse(self):
        return Orientation(np.linalg.inv(self.data))

    def ang_dist(self, other):
        r = self * other.inverse()
        trace_r = r.data[0, 0] + r.data[1, 1] + r.data[2, 2]
        return np.arccos((trace_r - 1) / 2)

    @property
    def data(self):
        return self._data

    @property
    def array(self):
        return self._data

    @property
    def matrix(self):
        return np.matrix(self._data)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self._data @ other.data)
        elif isinstance(other, Orientation):
            return Orientation(self._data @ other.data)
        elif isinstance(other, np.ndarray):
            if other.shape[0] == 3:
                return self.data @ other
            elif other.shape[1] == 3:
                return (self.data @ other.T).T
            else:
                raise ValueError("Array shape must be 3,x or x,3")
        else:
            raise ValueError()

    __matmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, Orientation):
            return False
        # FIXME; This is dead simple but can we make it more efficient?
        v = Vector(1, 2, 3)
        return self @ v == other @ v

    def to_quaternion(self):
        '''
        Returns w, x, y, z
        adapted from
        https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        '''

        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = self._data.flat
        # Fill only lower half of symmetric matrix
        K = np.array([[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0], [
            Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0
        ], [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q[0], q[1], q[2], q[3]

    @staticmethod
    def from_quaternion(w, x, y, z):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        Nq = w * w + x * x + y * y + z * z
        if Nq < float_eps:
            return np.eye(3)
        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z
        return Orientation(
            np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                      [xZ - wY, yZ + wX, 1.0 - (xX + yY)]]))

    @staticmethod
    def from_axis_angle(axis, angle, is_normalized=False):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        x, y, z = axis
        if not is_normalized:
            n = math.sqrt(x * x + y * y + z * z)
            x = x / n
            y = y / n
            z = z / n
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1 - c
        xs = x * s
        ys = y * s
        zs = z * s
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        return Orientation(
            np.array([[x * xC + c, xyC - zs, zxC + ys], [xyC + zs, y * yC + c, yzC - xs],
                      [zxC - ys, yzC + xs, z * zC + c]]))

    def to_axis_angle(self, unit_thresh=1e-5):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        M = np.asarray(self._data, dtype=np.float)
        # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
        L, W = np.linalg.eig(M.T)
        i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
        if i.size == 0:
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        direction = np.real(W[:, i[-1]]).squeeze()
        # rotation angle depending on direction
        cosa = (np.trace(M) - 1.0) / 2.0
        if abs(direction[2]) > 1e-8:
            sina = (M[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
        elif abs(direction[1]) > 1e-8:
            sina = (M[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
        else:
            sina = (M[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
        angle = math.atan2(sina, cosa)
        return Vector(direction), angle

    def rotation_vector(self, unit_thresh=1e-5):
        v, a = self.to_axis_angle(unit_thresh)
        return v * a

    def copy(self):
        return Orientation(self.data.copy())
