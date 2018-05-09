import m3d
import math3d  # as ref for testing
from IPython import embed


def test_init():
    t = m3d.Transform()
    t.pos.x == 0
    t.pos.y == 0
    t.pos.x = 2
    t.pos.x == 0
    
    i = t.inverse
    morten = math3d.Transform()
    morten.pos.x = 2
    morten.inverse == i
    embed()

if __name__ == "__main__":
    test_init()


