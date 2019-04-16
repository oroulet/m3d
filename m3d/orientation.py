import math

import numpy as np

from m3d.vector import Vector
from m3d.common import float_eps


class Orientation(object):
    def __init__(self, data: np.ndarray = None, dtype=np.float32):
        if isinstance(data, np.ndarray):
            if data.shape == (3, 3):
                self._data = data
            else:
                raise ValueError(f"A numpy array of size (3, 3) is expected not {data.shape}")
        elif isinstance(data, list):
            self._data = np.array(data)
            if self._data.shape != (3, 3):
                raise ValueError(f"Creating an array from argument {data} did not lead to an array of shape (3, 3)")
        elif data is None:
            self._data = np.identity(3, dtype=dtype)
        else:
            raise ValueError(f"A numpy array of size (3, 3) is expected not {data}")

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
        data = np.array2string(self.data, separator=', ')
        return "Orientation(\n{}\n)".format(data)

    __repr__ = __str__

    def inverse(self):
        return Orientation(np.linalg.inv(self.data))

    def ang_dist(self, other):
        r = self * other.inverse()
        trace_r = r.data[0, 0] + r.data[1, 1] + r.data[2, 2]
        if trace_r > 3:
            # might happen with approximations/rouding
            trace_r = 3
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
            return NotImplemented

    __matmul__ = __mul__

    def __eq__(self, other):
        return self.similar(other)

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

    @staticmethod
    def from_xy(x_vec, y_vec):
        """
        Generate a new Orientation from two vectors using x as reference
        """
        if not isinstance(x_vec, Vector):
            x_vec = Vector(x_vec)
        if not isinstance(y_vec, Vector):
            y_vec = Vector(y_vec)
        x_vec.normalize()
        y_vec.normalize()
        orient = Orientation()
        orient._data[:, 0] = x_vec.data
        orient._data[:, 2] = x_vec.cross(y_vec).normalized().data
        orient._data[:, 1] = Vector(np.cross(orient._data[:, 2], x_vec.data)).normalized().data
        return orient

    @staticmethod
    def from_yz(y_vec, z_vec):
        """
        Generate a new Orientation from two vectors using y as reference
        """
        if not isinstance(y_vec, Vector):
            y_vec = Vector(y_vec)
        if not isinstance(z_vec, Vector):
            z_vec = Vector(z_vec)
        y_vec.normalize()
        z_vec.normalize()
        orient = Orientation()
        orient._data[:, 1] = y_vec.data
        orient._data[:, 0] = y_vec.cross(z_vec).normalized().data
        orient._data[:, 2] = Vector(np.cross(orient._data[:, 0], y_vec.data)).normalized().data
        return orient

    @staticmethod
    def from_xz(x_vec, z_vec, ref='x'):
        """
        Generate a new Orientation from two vectors using x as reference
        """
        if not isinstance(x_vec, Vector):
            x_vec = Vector(x_vec)
        if not isinstance(z_vec, Vector):
            z_vec = Vector(z_vec)
        x_vec.normalize()
        z_vec.normalize()
        orient = Orientation()
        orient._data[:, 1] = z_vec.cross(x_vec).normalized().data

        if ref == 'x':
            orient._data[:, 0] = x_vec.data
            orient._data[:, 2] = Vector(np.cross(x_vec.data, orient._data[:, 1])).normalized().data
        elif ref == 'z':
            orient._data[:, 2] = z_vec.data
            orient._data[:, 0] = Vector(np.cross(orient._data[:, 1], z_vec.data)).normalized().data

        else:
            raise ValueError('Value of ref can only be x or z')

        return orient

    def to_axis_angle(self, unit_thresh=1e-4):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        M = np.asarray(self._data, dtype=np.float32)
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

    def to_rotation_vector(self, unit_thresh=1e-5):
        v, a = self.to_axis_angle(unit_thresh)
        return v * a

    @staticmethod
    def from_rotation_vector(v):
        if isinstance(v, (np.ndarray, list, tuple)):
            v = Vector(*v)
        if not isinstance(v, Vector):
            raise ValueError("Method take a Vector as argument")
        if (v.data == 0).all():
            return Orientation(np.identity(3, dtype=np.float32))
        u = v.normalized()
        idx = (u.data != 0).argmax()
        return Orientation.from_axis_angle(u, v[idx] / u[idx])

    def copy(self):
        return Orientation(self.data.copy())

    def similar(self, other, tol=float_eps):
        """
        Return True if angular distance to other Orientation is less than tol
        return False otherwise
        """
        if not isinstance(other, Orientation):
            raise ValueError("Expecting an Orientation object, received {} of type {}".format(other, type(other)))
        return self.ang_dist(other) <= tol

    @property
    def vec_x(self):
        return Vector(self._data[:, 0])

    @property
    def vec_y(self):
        return Vector(self._data[:, 1])

    @property
    def vec_z(self):
        return Vector(self._data[:, 2])
