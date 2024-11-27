import numpy as np  # type: ignore
import sympy as sp
import heyoka as hy


def find_lagrange_points(mu_val):
    """
    Solves the hamiltonian to find the Lagrange points.


    Parameters
    ----------
    mu_val: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system

    Returns
    -------
    list[float, tuple]
        A list of floats are returned [xL1, xL2, xL3, (xL4, yL4), (xL5, yL5)]
    """
    x, y, z, _, vy, _, mu = sp.symbols("x y z vx vy vz par[0]")
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f = sp.Matrix(6, 1, get_hamiltonian_state_derivative(H, conj_mom=False).tolist())
    lp = sp.solve(f[3].evalf(subs={mu: mu_val, vy: 0, y: 0, z: 0}), x)
    lp = [float(x) for x in lp]
    lp.append(tuple((0.5 - mu_val, np.sqrt(3) / 2)))
    lp.append((0.5 - mu_val, -np.sqrt(3) / 2))
    return lp


def convert_conj_mom_to_vel(state):
    """
    Converts conjugate momenta to velocities of a state vector.


    Parameters
    ----------
    state: np.ndarray
        A state vector [x, y, z, px, py, pz]

    Returns
    -------
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz]

    """
    vx = state[3] + state[1]
    vy = state[4] - state[0]
    vz = state[5]
    state[3:6] = np.array([vx, vy, vz])
    return state


def convert_vel_to_conj_mom(state):
    """
    Converts velocities to conjugate momenta of a state vector.


    Parameters
    ----------
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz]

    Returns
    -------
    state: np.ndarray
        A state vector [x, y, z, px, py, pz]

    """
    px = state[3] - state[1]
    py = state[4] + state[0]
    pz = state[5]
    state[3:6] = np.array([px, py, pz])
    return state


def get_u_bar(mu=None, position=None):
    """
    Gets the symbolic or numerical $\bar{U}$ expression as defined in Dynamical Systems ..., Koon (2022).


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    position: np.ndarray
        A position vector [x, y, z]

    Returns
    -------
    float
        if position is not None: Ubar
    sp.core.add.Add
        if position is None: Ubar expression

    """

    if position is None and mu is None:
        x, y, z, mu = sp.symbols("x y z par[0]")
        mu2 = mu
        mu1 = 1 - mu
        r1 = sp.sqrt((x + mu2) ** 2 + y**2 + z**2)
        r2 = sp.sqrt((x - mu1) ** 2 + y**2 + z**2)

        U = (-mu1 / r1) - (mu2 / r2) - (0.5 * mu1 * mu2)
        return -0.5 * (x**2 + y**2) + U

    elif position is not None and mu is not None:
        mu2 = mu
        mu1 = 1 - mu
        x, y, z = position
        r1 = np.sqrt((x + mu2) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - mu1) ** 2 + y**2 + z**2)

        U = -mu1 / r1 - mu2 / r2 - 0.5 * mu1 * mu2
        return -0.5 * (x**2 + y**2) + U
    else:
        raise RuntimeError("The input combination is not viable.")


def get_cr3bp_hamiltonian(mu=None, state=None, conj_mom=False):
    """
    Retrieves the numerical or symbolic CR3BP Hamiltonian.


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz] or [x, y, z, px, py, pz]
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.

    Returns
    -------
    sp.core.add.Add
        if conj_mom and state is None:
    float
        if conj_mom and state is not None and mu is not None:
    sp.core.add.Add
        if not conj_mom and state is None:
    float
        if not conj_mom and state is not None and mu is not None:

    """

    if conj_mom and state is None:
        x, y, z, px, py, pz, mu = sp.symbols("x y z px py pz par[0]")
        Ubar = get_u_bar()
        return 0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar
    elif conj_mom and state is not None and mu is not None:
        Ubar = get_u_bar(mu, state[0:3])
        x, y, z, px, py, pz = state
        return 0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar
    elif not conj_mom and state is None:
        x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
        Ubar = get_u_bar()
        return 0.5 * (vx**2 + vy**2 + vz**2) + Ubar
    elif not conj_mom and state is not None and mu is not None:
        Ubar = get_u_bar(mu, state[0:3])
        x, y, z, vx, vy, vz = state
        return 0.5 * (vx**2 + vy**2 + vz**2) + Ubar
    else:
        raise RuntimeError("The input is malformed")


def get_jacobi_integral(mu, state=None, conj_mom=False):
    """
    Retrieves the Jacobi integral as defined by Dynamical Systems ..., Koon (2022).


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz] or [x, y, z, px, py, pz]
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.

    Returns
    -------
    sp.core.add.Add
        if conj_mom and state is None:
    float
        if conj_mom and state is not None and mu is not None:
    sp.core.add.Add
        if not conj_mom and state is None:
    float
        if not conj_mom and state is not None and mu is not None:

    """
    E = get_cr3bp_hamiltonian(mu, state, conj_mom)
    return -2 * E


def get_current_symbols(expression):
    """
    Returns a dict of symbols of any sp.core.Expression (type to be verified)


    Parameters
    ----------
    expression: sp.core.Expression

    Returns
    -------
    dict
        Dict including all free symbols of the expression

    """
    if isinstance(expression, sp.core.expr.Expr):
        symbol_dict = {it: symb for it, symb in enumerate(expression.free_symbols)}
    else:
        raise RuntimeError("Type not implemented yet.")


def get_compiled_hamiltonian_state_derivative(conj_mom=False):
    """
    Retrieves the symbolic CR3BP Hamiltonian state derivative as a compiled Heyoka function.


    Parameters
    ----------
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.

    Returns
    -------
    hy.core.cfunc_dbl

    """
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f_dyn_sp = get_hamiltonian_state_derivative(H, conj_mom=conj_mom)

    f_dyn = []
    for i in range(6):
        f_dyn.append(hy.from_sympy(f_dyn_sp[i]))

    if conj_mom:
        symbols_state = ["x", "y", "z", "px", "py", "pz"]
    else:
        symbols_state = ["x", "y", "z", "vx", "vy", "vz"]
    x = np.array(hy.make_vars(*symbols_state))
    return hy.cfunc(f_dyn, vars=x)


def get_hamiltonian_state_derivative(H, mu_val=None, state=None, conj_mom=False):
    """
    Retrieves the symbolic CR3BP Hamiltonian state derivative.


    Parameters
    ----------
    H: sp.core.add.Add
        A symbolic expression for the hamiltonian
    mu_val: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz] or [x, y, z, px, py, pz]
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.

    Returns
    -------
    sp.Array
        if conj_mom and state is None: An array of symbolic expressions for the state derivative
    sp.Array
        elif not conj_mom and state is None: An array of symbolic expressions for the state derivative
    np.ndarray
        elif conj_mom and state is not None: The evaluated state derivative
    np.ndarray
        elif not conj_mom and state is not None: The evaluated state derivative

    """

    x, y, z, px, py, pz, mu = sp.symbols("x y z px py pz par[0]")
    xdot = sp.diff(H, px)
    ydot = sp.diff(H, py)
    zdot = sp.diff(H, pz)
    pxdot = -sp.diff(H, x)
    pydot = -sp.diff(H, y)
    pzdot = -sp.diff(H, z)
    f = sp.Array([xdot, ydot, zdot, pxdot, pydot, pzdot])
    if conj_mom and state is None:
        return f
    elif not conj_mom and state is None:
        x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
        f_vel = sp.Array(
            [
                state_element.subs(px, vx - y).subs(py, vy + x).subs(pz, vz)
                for state_element in f
            ]
        )
        xdot = f_vel[0]
        ydot = f_vel[1]
        zdot = f_vel[2]
        vxdot = vy + f_vel[3]
        vydot = -vx + f_vel[4]
        vzdot = f_vel[5]
        return sp.Array([xdot, ydot, zdot, vxdot, vydot, vzdot])
    elif conj_mom and state is not None:
        state_val_dict = {
            x: state[0],
            y: state[1],
            z: state[2],
            px: state[3],
            py: state[4],
            pz: state[5],
            mu: mu_val,
        }
        f_eval = np.zeros((len(f)))
        for it, state_item in enumerate(f):
            current_variables = list(state_item.free_symbols)
            f_eval[it] = state_item.subs(
                {symb: state_val_dict[symb] for symb in current_variables}
            ).evalf()
        return f_eval
    elif not conj_mom and state is not None:
        x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
        f_vel = sp.Array(
            [
                state_element.subs(px, vx - y).subs(py, vy + x).subs(pz, vz)
                for state_element in f
            ]
        )
        xdot = f_vel[0]
        ydot = f_vel[1]
        zdot = f_vel[2]
        vxdot = vy + f_vel[3]
        vydot = -vx + f_vel[4]
        vzdot = f_vel[5]
        f = sp.Array([xdot, ydot, zdot, vxdot, vydot, vzdot])
        return np.array(
            [
                state_item.evalf(
                    subs={
                        x: state[0],
                        y: state[1],
                        z: state[2],
                        vx: state[3],
                        vy: state[4],
                        vz: state[5],
                        mu: mu_val,
                    }
                )
                for state_item in f
            ],
            dtype="float64",
        )


def convert_sympy_to_hy(state_derivative):
    """
    Convert the state derivative from Sympy to Heyoka.


    Parameters
    ----------
    state_derivative: sp.Array
        A tuple, list, np.ndarray or list of tuples of Sympy symbolic expressions

    Returns
    -------
    list[hy.core.expression]
        A list of Heyoka expressions

    """
    if (
        isinstance(
            state_derivative,
            (tuple, list, np.ndarray),
        )
        and isinstance(state_derivative[0], sp.core.symbol.Symbol)
    ) or isinstance(
        state_derivative, sp.tensor.array.dense_ndim_array.ImmutableDenseNDimArray
    ):
        return [hy.from_sympy(i) for i in state_derivative]
    elif (
        isinstance(state_derivative, list)
        and isinstance(state_derivative[0], tuple)
        and isinstance(state_derivative[0][0], sp.core.symbol.Symbol)
    ):
        state_derivative_hy = []
        for tup in state_derivative:
            state_derivative_hy.append(tuple(hy.from_sympy(i) for i in tup))
        return state_derivative_hy
    else:
        raise RuntimeError(
            "The type provided is not implemented and can not be dealt with."
        )


def get_cr3bp_jacobian(mu_val=None, state=None, conj_mom=False):
    """
    Retrieves the symbolic or numerical Jacobian matrix of the CR3BP Hamiltonian.


    Parameters
    ----------
    mu_val: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    state: np.ndarray
        A state vector [x, y, z, vx, vy, vz] or [x, y, z, px, py, pz]
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.


    Returns
    -------
    np.ndarray
        if mu_val is None and state is None: A 6x6 array of Jacobian derivative values
    sympy.matrices.dense.MutableDenseMatrix
        if mu_val is not None and state is not None: The symbolic 6x6 array of the Jacobian

    """
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f = sp.Matrix(6, 1, get_hamiltonian_state_derivative(H, conj_mom=conj_mom).tolist())
    if not conj_mom:
        x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
        state_symbs = sp.Matrix([x, y, z, vx, vy, vz])
    else:
        x, y, z, px, py, pz, mu = sp.symbols("x y z px py pz par[0]")
        state_symbs = sp.Matrix([x, y, z, px, py, pz])
    jacobian = f.jacobian(state_symbs)
    if mu_val is None and state is None:
        return jacobian
    elif mu_val is not None and state is not None:
        jacobian = jacobian.evalf(
            subs={
                state_symbs[0]: state[0],
                state_symbs[1]: state[1],
                state_symbs[2]: state[2],
                state_symbs[3]: state[3],
                state_symbs[4]: state[4],
                state_symbs[5]: state[5],
                mu: mu_val,
            }
        )
        return np.array(jacobian).astype(np.float64)
    # elif mu_val is not None and state is None:
    #     jacobian.evalf(subs={mu: mu_val, x: , y: 0, z: 0, vx: 0, vy: 0, vz: 0})
    #     return np.array(jacobian).astype(np.float64)
    else:
        raise RuntimeError(
            "The inputs given do not allow for either a fully symbolic or a fully numerical result. Revisit inputs."
        )
