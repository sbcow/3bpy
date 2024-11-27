from src.richardson.brent import brent
from src.richardson.clp_func import clp_func


def xlp_coord(mu=None, ilp=1):
    """
    location of libration point normalized x-component
    input
    ilp = libration point of interest (0, 1 or 2)
    output
    xlp = normalized x-coordinate of libration point
    """

    # convergence criterion for Brent's method

    rtol = 1e-10
    # lower and upper bounds for coordinate search

    xlwr = -2.0
    xupr = 2.0
    if 1 == ilp:
        xlp, _ = brent(clp_func, xlwr, xupr, rtol, mu, ilp)
    elif 2 == ilp:
        xlp, _ = brent(clp_func, xlwr, xupr, rtol, mu, ilp)
    elif 3 == ilp:
        xlp, _ = brent(clp_func, xlwr, xupr, rtol, mu, ilp)

    return xlp
