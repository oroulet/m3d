import numpy as np

from m3d.vector import Vector
from m3d.orientation import Orientation


class Transform(object):
    """
    Create a new Transform object
    Accepts an orientation and a vector or a matrix 4*4 as argument
    Rmq:
    When creating a transform from a 4*4 Matrix, the matrix is directly used
    as the Transform data
    When accessing/modifying the Orientation or Vector object you are 
    modifying a vew of the matrix data
    When creating a new Transform object from an Orientation and 
    Vector or 2 numpy arrays, you are copying them
    """

    def __init__(self, orientation=None, vector=None, matrix=None, dtype=np.float32):
        if matrix is not None:
            self._data = matrix
        else:
            self._data = np.identity(4, dtype=dtype)
        if orientation is None:
            pass
        elif isinstance(orientation, np.ndarray):
            if orientation.shape == (3, 3):
                self._data[:3, :3] = orientation
            else:
                raise ValueError()
        elif isinstance(orientation, Orientation):
            self._data[:3, :3] = orientation.data
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._orient = Orientation(self._data[:3, :3], dtype=dtype)

        if vector is None:
            pass
        elif isinstance(vector, np.ndarray):
            self._data[:3, 3] = vector
        elif isinstance(vector, Vector):
            self._data[:3, 3] = vector.data
        elif isinstance(vector, (list, tuple)):
            self._data[:3, 3] = vector
        else:
            raise ValueError("orientation argument should be a numpy array, Orientation or None")
        self._pos = Vector(self._data[:3, 3], dtype=dtype)

    def __str__(self):
        return "Transform(\n{},\n{}\n)".format(self.orient, self.pos)

    __repr__ = __str__

    @property
    def pos(self):
        """
        Access the position part of the matrix through a Vector object
        """
        return self._pos

    @pos.setter
    def pos(self, vector):
        if not isinstance(vector, Vector):
            raise ValueError()
        self._data[:3, 3] = vector.data
        self._pos = Vector(self._data[:3, 3])  # make sure vector data is a view on our data

    @property
    def orient(self):
        """
        Access the orientation part of the matrix through an Orientation object
        """
        return self._orient

    @orient.setter
    def orient(self, orient):
        if not isinstance(orient, Orientation):
            raise ValueError()
        self._data[:3, :3] = orient.data
        self._orient = Orientation(self._data[:3, :3])  # make sure orientation data is view on our data

    @property
    def data(self):
        """
        Access the numpy array used by Transform
        """
        return self._data

    array = data

    @property
    def matrix(self):
        """
        Access the numpy array used by Transform as a numpy Matrix object
        """
        return np.matrix(self._data)

    def inverse(self):
        """
        Return inverse of Transform
        """
        return Transform(matrix=np.linalg.inv(self._data))

    def invert(self):
        """
        In-place inverse the matrix
        """
        self._data = np.linalg.inv(self._data)

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
            return Transform(matrix=self._data @ other.data)
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
        return self.to_pose_vector()

    def to_pose_vector(self):
        """
        Return a representation of transformation as 6 numbers array
        3 for position, and 3 for rotation vector
        """
        v = self.orient.to_rotation_vector()
        return np.array([self.pos.x, self.pos.y, self.pos.z, v.x, v.y, v.z])

    @staticmethod
    def from_pose_vector(x, y, z, r1, r2, r3):
        o = Orientation.from_rotation_vector(Vector(r1, r2, r3))
        return Transform(o, [x, y, z])

    def to_ros(self):
        return self.orient.to_quaternion(), self.pos.data

    @staticmethod
    def from_ros(q, v):
        orient = Orientation.from_quaternion(*q)
        return Transform(orient, Vector(v))

    def copy(self):
        return Transform(matrix=self._data.copy())

    def dist(self, other):
        """
        Return distance equivalent between this matrix and a second one
        """
        return self.pos.dist(other.pos) + self.orient.ang_dist(other.orient)
