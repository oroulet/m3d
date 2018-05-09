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
    #t.pos.x = 1
    t.orient.rotate_yb(1)
    m = math3d.Transform()
    m.orient.rotate_yb(1)
    embed()


if __name__ == "__main__":
    test_rotation()


