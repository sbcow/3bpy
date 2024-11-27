import numpy as np

def brent(f=None, x1=None, x2=None, rtol=None, mu=None, ilp=1):
    """
    solve for a single real root of a nonlinear equation
    Brent's method
    input
    f    = objective function coded as y = f(x)
    x1   = lower bound of search interval
    x2   = upper bound of search interval
    rtol = algorithm convergence criterion
    output
    xroot  = real root of f(x) = 0
    froot  = function value at f(x) = 0
    """

    # machine epsilon
    eps = 2.23e-16
    e = 0
    a = x1
    b = x2
    fa = f(a, mu, ilp)
    fb = f(b, mu, ilp)
    fc = fb
    for iter in np.arange(1, 50 + 1, 1).reshape(-1):
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d
        if np.abs(fc) < np.abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol1 = 2.0 * eps * np.abs(b) + 0.5 * rtol
        xm = 0.5 * (c - b)
        if np.abs(xm) <= tol1 or fb == 0:
            break
        if np.abs(e) >= tol1 and np.abs(fa) > np.abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = np.abs(p)
            min = np.abs(e * q)
            tmp = 3.0 * xm * q - np.abs(tol1 * q)
            if min < tmp:
                min = tmp
            if 2.0 * p < min:
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if np.abs(d) > tol1:
            b = b + d
        else:
            b = b + np.sign(xm) * tol1
        fb = f(b, mu, ilp)

    xroot = b
    froot = fb
    return xroot, froot
