import numpy as np

from m3d.common import float_eps
from m3d.quaternion import Quaternion


class DualQuaternion:
    def __init__(self, vector=None):
        if vector is not None:
            self._data = vector

    def __str__(self):
        return "DualQuaternion(\n{}, \n{})".format(self._data[:4], self._data[4:])

    __repr__ = __str__

    @property
    def data(self):
        return self._data

    @property
    def q_rot(self):
        return Quaternion(self._data[:4])

    @property
    def q_trans(self):
        return Quaternion(self._data[4:])

    @staticmethod
    def from_transformation(trans):
        """
        Write the transformation as a dual quaternion. A dual quaternion can be
        written as q = r + epslion * d
        where the r quaternion is known as the real or rotational part and the
        d quaternion is known as the dual or displacement part. epsilon is the
        dual unit.
        We denote q_rot as the rotation part and q_trans as the displacement
        part.
        """

        q_rot = Quaternion(trans.orient.to_quaternion())
        q_trans = 0.5 * Quaternion([0, *trans.pos.data]) * q_rot
        return DualQuaternion([*q_rot.data, *q_trans.data])

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            q_rot = self.q_rot * other.q_rot
            q_trans = self.q_rot * other.q_trans + self.q_trans * other.q_rot
            return DualQuaternion([*q_rot.data, *q_trans.data])
        if isinstance(other, (float, int)):
            return DualQuaternion(other * self._data)
        raise ValueError()

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return DualQuaternion(self._data * other)
        raise ValueError()

    @property
    def conjugate(self):
        return DualQuaternion([*self.q_rot.conjugate.data, *self.q_trans.conjugate.data])

    @property
    def unity_condition(self):
        """
        When working with rigid rotational and translational transformations
        the dual quaternion must fulfill the unit condition, i.e.
        it must be of unit length.
        """
        norm = np.linalg.norm((self * self.conjugate).data)
        return abs(1 - norm) < float_eps
