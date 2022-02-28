import mpmath as mpm

from hyperlib.utils.multiprecision import * 

TOL = 1e-25
mpm.mp.dps = 30 

def test_poincare_dist():
    x = mpm.matrix([[0.,0.,-0.4]])
    y = mpm.matrix([[0.0,0.0,0.1]])
    o = mpm.zeros(1,3)

    assert poincare_dist(o,y)-poincare_dist0(y) < TOL
    assert poincare_dist(o,x)-poincare_dist0(x) < TOL 
    assert poincare_dist(o,x)+poincare_dist(o,y)-poincare_dist(x,y) < TOL

def test_poincare_reflect0():
    z = mpm.matrix([[0.,0.,0.5]])
    x = mpm.matrix([[0.1,-0.2,0.1]])

    assert mpm.norm( poincare_reflect0(z, z) ) < TOL 

    y = poincare_reflect0(z,x)
    assert mpm.norm( poincare_reflect0(z,y) - x ) < TOL 

def test_poincare_reflect():
    x = mpm.matrix([[0.3,-0.3,0.0]])
    a = mpm.matrix([[0.5,0.,0.0]])
    y = poincare_reflect(a, x)
    
    R = mpm.norm(a)
    r1 = mpm.norm(x-a)
    r2 = mpm.norm(y-a)
    assert R**2 - r1*r2 < TOL 
