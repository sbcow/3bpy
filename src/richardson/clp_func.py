def clp_func(x=None, mu=None, ilp=1):
    """
    normalized x-coordinate function
    input
    x = current argument
    output
    y = objective function evaluated at x
    global
    ilp = libration point number (1, 2 or 3)
    mu  = normalized gravity constant
    """

    if 1 == ilp:
        y = x - (1.0 - mu) / (mu + x) ** 2 + mu / (x - 1.0 + mu) ** 2
    elif 2 == ilp:
        y = x - (1.0 - mu) / (mu + x) ** 2 - mu / (x - 1.0 + mu) ** 2
    elif 3 == ilp:
        y = x + (1.0 - mu) / (mu + x) ** 2 + mu / (x - 1.0 + mu) ** 2

    return y
