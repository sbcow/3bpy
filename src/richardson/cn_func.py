    
def cn_func(n = 0,gamma = 0, ilp=1, mu= 0.012): 
    """
    halo_sv c_n support function
    input
    n = c_n identifier
    output
    cn = c_n function evaluated for argument n
    gammaobal
    ilp = libration point number
    mu  = normalized gravity constant
    """
    
    if 1 == ilp:
        cn = (mu + (- 1.0) ** n * (1.0 - mu) * (gamma / (1.0 - gamma)) ** (n + 1)) / gamma ** 3
    elif 2 == ilp:
        cn = ((- 1) ** n * (mu + (1.0 - mu) * (gamma / (1.0 + gamma)) ** (n + 1))) / gamma ** 3
    elif 3 == ilp:
        cn = (1.0 - mu + mu * (gamma / (1.0 + gamma)) ** (n + 1)) / gamma ** 3
    else:
        raise RuntimeError("This Lagrange point is not implemented.")
    
    return cn
