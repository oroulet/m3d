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
    assert np.array_equal(m._data, t.data)


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


if __name__ == "__main__":
    test_transform()


