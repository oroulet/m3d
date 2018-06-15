import math3d  # as ref for testing
import numpy as np
from IPython import embed


import m3d


def _are_equals(m1, m2):
    """
    to test equality of math3d and m3d vectors
    """

    return (m1 - m2).mean() < m3d.float_eps


def test_init():
    t = m3d.Transform()
    assert t.pos.x == 0
    assert t.pos.y == 0
    t.pos.x = 2
    assert t.pos.x == 2

    i = t.inverse()
    morten = math3d.Transform()
    morten.pos.x = 2
    assert np.array_equal(morten.inverse._data, i.data)


def test_rotation():
    t = m3d.Transform()
    t.pos.x = 1
    t.orient.rotate_yb(1)
    m = math3d.Transform()
    m.orient.rotate_yb(1)
    m.pos.x = 1
    print(t)
    print(m)
    assert _are_equals(m._data, t.data)


def test_multiplication_orient():
    o = m3d.Orientation()
    o.rotate_zb(np.pi / 2)
    v = m3d.Vector(1, 0, 0)
    r = o * v
    assert r == m3d.Vector(0, 1, 0)
    o.rotate_zb(-np.pi)
    v = m3d.Vector(2, 0, 0)
    r = o * v
    assert r == m3d.Vector(0, -2, 0)


def test_transform():
    t = m3d.Transform()
    t.orient.rotate_zb(np.pi/2)
    t.pos.y = 2
    v = m3d.Vector(1, 0, 0)
    r = t * v

    assert r == m3d.Vector(0, 3, 0)


def test_pose_vector():
    t = m3d.Transform()
    t.pos.x = 1
    t.pos.z = 2
    t.orient.rotate_yb(1)
    m = math3d.Transform()
    m.orient.rotate_yb(1)
    m.pos.x = 1
    m.pos.z = 2
    assert _are_equals(t.pose_vector, m.pose_vector)
    t.orient.rotate_zb(2)
    m.orient.rotate_zb(2)
    assert _are_equals(t.pose_vector, m.pose_vector)
    t.orient.rotate_xb(-2)
    m.orient.rotate_xb(-2)
    assert _are_equals(t.pose_vector, m.pose_vector)


def test_mult_trans():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi/2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi/2)
    t2.pos.x = 2

    v = m3d.Vector(0, 0, 3)

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    tm = math3d.Transform()
    tm.orient.rotate_xb(np.pi/2)
    tm.pos.x = 2
    vm = math3d.Vector(0, 0, 3)

    assert t1 * t2 * v == tr * v
    assert t1 @ t2 @ v == tr @ v
    assert _are_equals((t2 * v).data, (tm * vm)._data)


def test_equal():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi/2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi/2)
    t2.pos.x = 2

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    assert t1 != t2
    assert t1 != tr
    assert t1 * t2 == tr
    assert t2 * t1 == tr


def test_inverse():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi / 3)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(-13 * np.pi / 6)
    t2.pos.x = 2.3

    v = m3d.Vector(0.1, -4.5, 3.0)

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    assert (t1 * t1.inverse()) == m3d.Transform(matrix=np.identity(4))
    assert (t2 * t2.inverse()) == m3d.Transform(matrix=np.identity(4))
    assert (t1 * t2 * t1.inverse() * t2.inverse()) == m3d.Transform(matrix=np.identity(4))
    assert t1.inverse() * (t1 * v) == v



def test_construct():
    o = m3d.Orientation()
    o.rotate_zb(1)
    v = m3d.Vector()
    v[0] = 1
    v[2] = 2
    t = m3d.Transform(o, v)
    assert t.pos.x == 1
    assert t.pos.z == 2
    t.pos = m3d.Vector()
    t.orient.rotate_zb(-1)
    assert t == m3d.Transform(matrix=np.identity(4))
    t.orient = o
    assert t != m3d.Transform(matrix=np.identity(4))


def test_orient():
    o = m3d.Orientation()
    o.rotate_zb(2)
    o2 = m3d.Orientation()
    o2.rotate_zb(2 * np.pi)
    assert o * o2 == o


def test_quaternion():
    o = m3d.Orientation()
    o.rotate_xb(np.pi / 3)
    o.rotate_zb(np.pi / 3)
    q = o.to_quaternion()
    o2 = m3d.Orientation.from_quaternion(*q)
    assert o == o2


def test_axis_angle():
    o = m3d.Orientation()
    o.rotate_xb(np.pi / 3)
    o.rotate_zb(np.pi / 3)
    v, a = o.to_axis_angle()
    o2 = m3d.Orientation.from_axis_angle(v, a)
    assert o == o2


def test_pc():
    pc = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 5]])
    t = m3d.Transform()
    t.pos.x = 1.2
    t.pos.y = 1
    t.orient.rotate_yb(1)
    t.orient.rotate_zb(1)

    tm = math3d.Transform()
    tm.pos.x = 1.2
    tm.pos.y = 1
    tm.orient.rotate_yb(1)
    tm.orient.rotate_zb(1)

    assert _are_equals(t.data, tm._data)
    a = t * pc
    b = tm * pc
    assert _are_equals(a, b)
    assert a.shape == pc.shape
    
    c = t * pc.T
    assert c.shape == pc.T.shape


def test_copy():
    t = m3d.Transform()
    t.pos.x = 1
    new = t.copy()
    t.orient.rotate_zb(1)
    new.pos.x = 5
    assert t.pos.x != new.pos.x
    assert t.orient.data[0, 0] != new.orient.data[0, 0]


def test_substraction():
    v1 = m3d.Vector(1, 2, 3)
    v3 = m3d.Vector(2, -3, 4)
    v_res = v3 - v1
    assert v_res == m3d.Vector(1, -5, 1)


def test_addition():
    v1 = m3d.Vector(1, 2, 3)    
    v3 = m3d.Vector(2, -3, 4)
    v_res = v3 + v1
    assert v_res == m3d.Vector(3, -1, 7)


if __name__ == "__main__":
    test_pc()


