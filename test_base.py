import math3d  # as ref for testing
import numpy as np
from IPython import embed

import m3d


def _are_equals(m1, m2, eps=m3d.float_eps):
    """
    to test equality of math3d and m3d vectors
    """
    m1 = _to_np(m1)
    m2 = _to_np(m2)

    print("DIST", abs(m1 - m2))
    return (abs(m1 - m2) <= eps).all()


def _to_np(obj):
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (m3d.Vector, m3d.Orientation, m3d.Transform)):
        return obj.data
    else:
        raise ValueError("Could not convert ob to nupy array", obj, type(obj))


def test_init():
    t = m3d.Transform()
    assert t.pos.x == 0
    assert t.pos.y == 0
    t.pos.x = 2
    assert t.pos.x == 2

    i = t.inverse()
    assert m3d.Vector(-2, 0, 0) == i.pos
    assert m3d.Orientation() == i.orient


def test_rotation():
    t = m3d.Transform()
    t.pos.x = 1
    t.orient.rotate_yb(1)
    res = m3d.Transform(
        m3d.Orientation([[0.54030228, 0, 0.84147096], [0., 1, 0.], [-0.84147096, 0., 0.54030228]]), m3d.Vector(1, 0, 0))
    assert _are_equals(res.data, t.data)


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
    t.orient.rotate_zb(np.pi / 2)
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


def test_vec_eq():
    a = m3d.Vector(-1.0000001192092896, -1.9999998807907104, -3.0)
    b = m3d.Vector(-1.0, -2.0, -3.0)
    assert a == b


def test_mult_trans():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi / 2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi / 2)
    t2.pos.x = 2

    v = m3d.Vector(0, 0, 3)

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    tm = math3d.Transform()
    tm.orient.rotate_xb(np.pi / 2)
    tm.pos.x = 2
    vm = math3d.Vector(0, 0, 3)

    assert t1 * t2 * v == tr * v
    assert t1 @ t2 @ v == tr @ v
    assert _are_equals((t2 * v).data, (tm * vm)._data)


def test_equal():
    t1 = m3d.Transform()
    t1.orient.rotate_xb(np.pi / 2)
    t1.pos.x = 1

    t2 = m3d.Transform()
    t2.orient.rotate_xb(np.pi / 2)
    t2.pos.x = 2

    tr = m3d.Transform()
    tr.orient.rotate_xb(np.pi)
    tr.pos.x = 3

    assert t1 != t2
    assert t1 != tr
    assert t1 * t2 == tr
    assert t2 * t1 == tr


def test_inverse_orient():
    o = m3d.Orientation()
    o.rotate_xb(3)
    o.rotate_yb(1)
    o.rotate_xb(1)
    v = m3d.Vector(-1, -2, -3)
    assert o * v != v
    assert o.inverse() * o * v == v
    assert o * o.inverse() * v == v
    assert o * o.inverse() * o == o


def test_inverse_trans():
    t = m3d.Transform()
    t.pos.x = 1
    t.orient.rotate_zb(np.pi / 2)
    t.orient.rotate_yb(np.pi / 2)
    v = m3d.Vector(2, 0, 0)
    assert t * v == m3d.Vector(1, 2, 0)


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
    assert (t1 * t1.inverse()) == m3d.Transform()
    assert (t2 * t2.inverse()) == m3d.Transform(matrix=np.identity(4))
    assert (t2 * t2.inverse()) == m3d.Transform()
    assert (t1 * t2 * t1.inverse() * t2.inverse()) == m3d.Transform(matrix=np.identity(4))
    assert t1.inverse() * (t1 * v) == v


def test_inverse_2():
    t = m3d.Transform()
    t.pos.x = 1
    t.pos.z = 3
    t.orient.rotate_yb(1)
    t.orient.rotate_zb(1)
    v = m3d.Vector(2, 3, 4)
    assert v != t * v
    assert v == t.inverse() * t * v
    assert v == t * t.inverse() * v
    assert v != t @ v
    assert v == t.inverse() @ t @ v
    assert v == t @ t.inverse() @ v


def test_rotation_seq():
    t = m3d.Transform()
    t.pos.x = 1
    t.pos.z = 3
    t.orient.rotate_xb(1)
    res = t.copy()
    t.orient.rotate_yb(2)
    t.orient.rotate_zb(3)
    t.orient.rotate_zb(-3)
    t.orient.rotate_yb(-2)
    assert t == res


def test_rotation_seq_2():
    t = m3d.Transform()
    t.pos.x = 1
    t.pos.z = 3
    t.orient.rotate_xb(1)
    t.orient.rotate_yb(2)
    t.orient.rotate_zb(3)

    b = m3d.Transform()
    b.orient.rotate_zb(-3)
    b.orient.rotate_yb(-2)
    b.orient.rotate_xb(-1)
    b.pos = b.orient * m3d.Vector(1, 0, 3) * -1

    assert _are_equals(t.inverse().data, b.data)


def test_rotation_t():
    t = m3d.Transform()
    t.pos.x = 1
    t.orient.rotate_zt(np.pi / 2)
    t.orient.rotate_yt(np.pi / 2)
    v = m3d.Vector(2, 0, 0)
    assert t * v == m3d.Vector(1, 0, -2)


def test_rotation_t_2():
    t = m3d.Transform()
    t.orient.rotate_yt(np.pi / 2)
    t.orient.rotate_zt(np.pi / 2)
    v = m3d.Vector(2, 0, 0)
    assert t * v == m3d.Vector(0, 2, 0)


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

    o_math = math3d.Orientation()
    o_math.rotate_xb(np.pi / 3)
    o_math.rotate_zb(np.pi / 3)
    q_math = o_math.get_unit_quaternion()
    o2_math = math3d.Orientation(q_math)
    assert _are_equals(np.array(q), np.array((q_math.s, q_math.x, q_math.y, q_math.z)))


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
    assert _are_equals(a[0], b[0])
    assert _are_equals(a[-1], b[-1])
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
    v2 = m3d.Vector(2, -3, 4)
    v_res = v2 - v1
    assert v_res == m3d.Vector(1, -5, 1)


def test_addition():
    v1 = m3d.Vector(1, 2, 3)
    v2 = m3d.Vector(2, -3, 4)
    v_res = v2 + v1
    assert v_res == m3d.Vector(3, -1, 7)


def test_dist():
    v1 = m3d.Vector(1, 1, 1)
    v2 = m3d.Vector(2, 2, 2)
    v_res = v2.dist(v1)
    assert v_res == m3d.Vector(1, 1, 1).length


def test_ang_dist():
    o1 = m3d.Orientation()
    o1.rotate_yb(1)
    o1.rotate_xb(1)
    o1.rotate_zb(1)

    o2 = m3d.Orientation()
    o2.rotate_yb(2)
    o2.rotate_xb(2)
    o2.rotate_zb(2)

    o_dist_m3d = o1.ang_dist(o2)
    o_inv_dist_m3d = o2.ang_dist(o1)

    o1_m = math3d.Orientation()
    o1_m.rotate_yb(1)
    o1_m.rotate_xb(1)
    o1_m.rotate_zb(1)

    o2_m = math3d.Orientation()
    o2_m.rotate_yb(2)
    o2_m.rotate_xb(2)
    o2_m.rotate_zb(2)

    o_dist_math3d = o1_m.ang_dist(o2_m)
    o_inv_dist_math3d = o2_m.ang_dist(o1_m)

    print('o_inv_dist_m3d', o_inv_dist_m3d)
    print('o_inv_dist_math3d', o_inv_dist_math3d)

    print('o_dist_m3d', o_dist_m3d)
    print('o_dist_math3d', o_dist_math3d)

    assert abs(o_dist_m3d - o_dist_math3d) <= m3d.float_eps
    assert abs(o_inv_dist_m3d - o_inv_dist_math3d) <= m3d.float_eps


def test_trans_dist():
    a = m3d.Transform()
    b = m3d.Transform()
    b.pos.x = 3
    b.orient.rotate_zb(1)
    b.orient.rotate_yb(1)
    assert a.dist(b) > 0.1
    assert b.dist(b) == 0


def test_eq():
    t = m3d.Transform()
    t.orient.rotate_yb(1)
    t.orient.rotate_zb(1)
    t.pos.x = 1
    v = m3d.Vector()
    o = m3d.Orientation()
    assert t != v
    assert v != t
    assert o != v
    assert v != o
    assert o != t
    assert t != o
    t2 = t.copy()
    assert t == t2
    assert t.pos == t2.pos
    assert t.orient == t2.orient
    t2.pos.y = 2
    assert t != t2
    assert t.pos != t2.pos
    assert t.orient == t2.orient
    t3 = t.copy()
    t3.orient.rotate_xb(1)
    assert t2.pos != t3.pos
    assert t2.orient != t3.orient
    assert t != t3


def test_norm():
    v = m3d.Vector(1, 2, 3)
    v.normalize()
    assert abs(v.length - 1) <= m3d.float_eps


def test_from_xz():
    x = m3d.Vector(1, 0, 0)
    z = m3d.Vector(0, 0, 1)
    orient = m3d.Orientation.from_xz(x, z)
    assert _are_equals(orient.data, np.identity(3))

    x = m3d.Vector(2, 0, 0.1)
    z = m3d.Vector(0.1, -0.1, 3)
    o = m3d.Orientation()
    o.rotate_yb(1)
    o.rotate_zb(1)
    x = o @ x
    z = o @ z
    orient = m3d.Orientation.from_xz(x, z)
    assert _are_equals(o, orient, eps=0.1)


if __name__ == "__main__":
    test_pc()
