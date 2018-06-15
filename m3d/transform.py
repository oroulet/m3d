import math

import numpy as np


float_eps = np.finfo(np.float32).eps


class Vector(object):
    """
    Represent a vector. 
    Takes either x, y, z as argument or an array
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
        return abs((self.data - other.data).mean()) < float_eps

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


class Orientation(object):
    def __init__(self, data: np.ndarray=None, dtype=np.float32):
        if isinstance(data, np.ndarray):
            self._data = data
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


class Transform(object):
    """
    Create a Transform
    Accepts an orientation and a vector or a matrci 4*4 as argument
    """
    def __init__(self, orientation=None, vector=None, matrix=None, dtype=np.float32):
        if matrix is not None:
            self.data = matrix
        else:
            self.data = np.identity(4, dtype=dtype)
        if orientation is None:
            pass
        elif isinstance(orientation, np.ndarray):
            if orientation.shape == (3, 3):
                self.data[:3, :3] = orientation
            elif orientation.shape == (4, 4):
                #FIXME: This crappy to take the orientation argument as a Transform...
                self.data = orientation
            else:
                raise ValueError()
        elif isinstance(orientation, Orientation):
            self.data[:3, :3] = orientation.data
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._orient = Orientation(self.data[:3, :3], dtype=dtype)

        if vector is None:
            pass
        elif isinstance(vector, np.ndarray):
            self.data[:3, 3] = vector
        elif isinstance(vector, Vector):
            self.data[:3, 3] = vector.data
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._pos = Vector(self.data[:3, 3], dtype=dtype)

    def __str__(self):
        return "Transform(\n{},\n{}\n)".format(self.orient, self.pos)

    __repr__ = __str__

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, vector):
        if not isinstance(vector, Vector):
            raise ValueError()
        self.data[:3, 3] = vector.data
        self._pos = Vector(self.data[:3, 3])  # make sure vector data is a view on our data

    @property
    def orient(self):
        return self._orient

    @orient.setter
    def orient(self, orient):
        if not isinstance(orient, Orientation):
            raise ValueError()
        self.data[:3, :3] = orient.data
        self._orient = Orientation(self.data[:3, :3])  # make sure orientation data is view on our data

    @property
    def array(self):
        return self.data

    @property
    def matrix(self):
        return np.matrix(self.data)

    def inverse(self):
        return Transform(np.linalg.inv(self.orient.data), -self.pos.data)

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        # FIXME; This is dead simple but can we make it more efficient?
        v = Vector([1, 2, 3])
        return self @ v == other @ v

    def __mul__(self, other):
        if isinstance(other, Vector):
            data = self.orient.data @ other.data + self.pos.data
            return Vector(data)
        elif isinstance(other, Transform):
            return Transform(matrix=self.data @ other.data)
        elif isinstance(other, np.ndarray):
            # This make it easy to support several format of point clouds but might be mathematically wrong
            if other.shape[0] == 3:
                return (self.orient.data @ other) + self.pos.data.reshape(3, 1)
            elif other.shape[1] == 3:
                return (self.orient.data @ other.T).T + self.pos.data
            else:
                raise ValueError("Array shape must be 3, x or x, 3")
        else:
            raise ValueError()

    __matmul__ = __mul__

    @property
    def pose_vector(self):
        v = self.orient.rotation_vector()
        return np.array([self.pos.x, self.pos.y, self.pos.z, v.x, v.y, v.z])

    def to_ros(self):
        return self.orient.to_quaternion(), self.pos.data
    
    @staticmethod
    def from_ros(q, v):
        orient = Orientation.from_quaternion(q)
        return Transform(orient, Vector(v))

    def copy(self):
        return Transform(self.data.copy())

