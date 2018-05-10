import numpy as np
import math

float_eps = np.finfo(np.float).eps


class Vector(object):
    def __init__(self, data=None):
        if isinstance(data, (list, tuple)):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data
        elif data is None:
            self._data = np.array([0, 0, 0])
        else:
            raise ValueError()

    @property
    def x(self):
        return self._data[0]

    @x.setter
    def x(self, val):
        self._data[0] = val

    @property
    def y(self):
        return self._data[1]

    @y.setter
    def y(self, val):
        self._data[1] = val

    @property
    def z(self):
        return self._data[2]

    @z.setter
    def z(self, val):
        self._data[2] = val

    def __str__(self):
        return "Vector({}, {}, {})".format(self.x, self.y, self.z)

    __repr__ = __str__

    @property
    def data(self):
        return self._data

    def __eq__(self, other):
        return (self.data - other.data).mean() < float_eps

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(self._data * other)
        else:
            raise ValueError()


class Orientation(object):
    def __init__(self, data=None):
        if isinstance(data, np.ndarray):
            self._data = data
        elif data is None:
            self._data = np.identity(3)
        else:
            raise ValueError()

    def rotate_xb(self, val):
        t = np.array([[1, 0, 0], [0, np.cos(val), -np.sin(val)], [0, np.sin(val), np.cos(val)]])
        self._data[:] = t @ self._data

    def rotate_yb(self, val):
        t = np.array([[np.cos(val), 0, np.sin(val)], [0, 1, 0], [-np.sin(val), 0, np.cos(val)]])
        self._data[:] = t @ self._data

    def rotate_zb(self, val):
        t = np.array([[np.cos(val), -np.sin(val), 0], [np.sin(val), np.cos(val), 0], [0, 0, 1]])
        self._data[:] = t @ self._data

    def __str__(self):
        return "Orientation(\n{}\n)".format(self.data)

    __repr__ = __str__

    @property
    def inverse(self):
        return Orientation(np.linalg.inv(self.data))

    @property
    def data(self):
        return self._data

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self._data @ other.data)
        elif isinstance(other, Orientation):
            return Orientation(self._data @ other.data)
        else:
            raise ValueError()

    def __eq__(self, other):
        # might be iunterestingt to use quaternion here or multiply a vector and compare result
        raise NotImplementedError()

    def to_quaternion(self):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
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
        return q

    @staticmethod
    def from_quaternion(self, q):
        # adapted from
        # https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
        w, x, y, z = q
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
    def from_axis_angle(self, axis, angle, is_normalized=False):
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
        if not len(i):
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
        v, a = self.to_axis_angle()
        return v * a


class Transform(object):
    def __init__(self, orientation=None, vector=None, matrix=None):
        if matrix is not None:
            self.data = matrix
        else:
            self.data = np.identity(4)
        if orientation is None:
            pass
        elif isinstance(orientation, np.ndarray):
            self.data[:3, :3] = orientation
        elif isinstance(orientation, Orientation):
            self.data[:3, :3] = orientation.data
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._orient = Orientation(self.data[:3, :3])

        if vector is None:
            pass
        elif isinstance(vector, np.ndarray):
            self.data[:3, 3] = vector
        elif isinstance(vector, Vector):
            self.data[:3, 3] = vector.data
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._pos = Vector(self.data[:3, 3])

    def __str__(self):
        return "Transform(\n{},\n{}\n)".format(self.orient, self.pos)

    __repr__ = __str__

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, vector):
        self.data[:3, 3] = vector.data
        self._pos = Vector(self.data[:3, 3])

    @property
    def orient(self):
        return self._orient

    @orient.setter
    def orient(self, orient):
        self.data[:, 3] = orient.data
        self._pos = Orientation(self.data[:, 3])

    @property
    def inverse(self):
        return Transform(np.linalg.inv(self.orient.data), -self.pos.data)

    def __eq__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Vector):
            data = self.orient.data @ other.data + self.pos.data
            return Vector(data)
        elif isinstance(other, Transform):
            return Transform(self.data @ other.data)
        else:
            raise ValueError()

    @property
    def pose_vector(self):
        v = self.orient.to_rotation_vector()
        return np.array([self.pos.x, self.pos.y, self.pos.z, v.x, v.y, v.z])
