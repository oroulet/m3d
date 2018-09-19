

[![pipeline status](https://gitlab.com/kurant/m3d/badges/master/pipeline.svg)](https://gitlab.com/kurant/m3d/commits/master)
[![coverage report](https://gitlab.com/kurant/m3d/badges/master/coverage.svg)](https://gitlab.com/kurant/m3d/commits/master)

Object oriented transformation matrix library. m3d was originally developed as an MIT licenced API compatible replacement for pymath3d which is GPL, but is now developed independently. Development is happenig here: https://gitlab.com/kurant-open/m3d/

m3d uses some code from https://matthew-brett.github.io/transforms3d/ and its API is inspired by pymath3d API from Morten Lind https://gitlab.com/morlin/pymath3d.


Example use:

    t = m3d.Transform()
    t.orient.rotate_yb(1)
    t2 = t.copy()
    t2.pos.x = 1
    v = t2.inverse() @ t @ m3d.Vector(1, 2, 3)
    v.normalize()
    v *= 1.2
    o = m3d.Orientation.from_axis_angle(0, 0, 1)
    v = o * v
    o2 = m3d.Orientation.from_xy(m3d.Vector(1.1, 0.2, 0), m3d.Vector(0, 0, 1.1))
    v = o2 * o.inverse() * v

