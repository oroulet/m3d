import m3d
import math3d  # as ref for testing
import numpy as np
from IPython import embed


def test_init():
    t = m3d.Transform()
    assert t.pos.x == 0
    assert t.pos.y == 0
    t.pos.x = 2
    assert t.pos.x == 2

    i = t.inverse
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
    assert (m._data - t.data).mean() < m3d.float_eps


def test_multiplication_orient():
    o = m3d.Orientation()
    o.rotate_zb(np.pi / 2)
    v = m3d.Vector((1, 0, 0))
    r = o * v
    assert r == m3d.Vector((0, 1, 0))
    o.rotate_zb(-np.pi)
    v = m3d.Vector((2, 0, 0))
    r = o * v
    assert r == m3d.Vector((0, -2, 0))


def test_transform():
    t = m3d.Transform()
    t.orient.rotate_zb(np.pi/2)
    t.pos.y = 2
    v = m3d.Vector((1, 0, 0))
    r = t * v

    assert r == m3d.Vector((0, 3, 0))


def test_pose_vector():
    t = m3d.Transform()
    t.pos.x = 1
    t.orient.rotate_yb(1)
    m = math3d.Transform()
    m.orient.rotate_yb(1)
    m.pos.x = 1
    assert (t.pose_vector - m.pose_vector).mean() < m3d.float_eps
    t.orient.rotate_zb(2)
    m.orient.rotate_zb(2)
    assert (t.pose_vector - m.pose_vector).mean() < m3d.float_eps
    t.orient.rotate_xb(-2)
    m.orient.rotate_xb(-2)
    assert (t.pose_vector - m.pose_vector).mean() < m3d.float_eps


def test_mult_trans():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi/2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi/2)
    t2.pos.x = 2

    v = m3d.Vector([0, 0, 3])

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    assert t1 * t2 * v == tr * v
    assert t1 @ t2 @ v == tr @ v


def test_equal():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi/2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi/2)
    t2.pos.x = 2

    v = m3d.Vector([0, 0, 3])

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

    v = m3d.Vector([0.1, -4.5, 3.0])

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    assert (t1 * t1.inverse) == m3d.Transform(matrix=np.identity(4))
    assert (t2 * t2.inverse) == m3d.Transform(matrix=np.identity(4))
    assert (t1 * t2 * t1.inverse * t2.inverse) == m3d.Transform(matrix=np.identity(4))
    assert t1.inverse * (t1 * v) == v






if __name__ == "__main__":
    test_equal()


