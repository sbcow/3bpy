import sympy as sp
import heyoka as hy

from src.hamiltonian_operations import (
    get_hamiltonian_state_derivative,
    convert_sympy_to_hy,
    get_cr3bp_hamiltonian,
)


def get_ta(ic, batch_mode=False, conj_mom=False):
    """
    Retrieves a Taylor-adaptive integrator.

    Parameters
    ----------
    ic: np.ndarray or list[np.ndarray]
        Initial state used for initializing the integrator
    batch_mode: bool
        Determines whether multiple initial states are to be given propagated simultaneously
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.


    Returns
    -------
    hy.taylor_adaptive
        A Taylor adaptive integrator
    """
    if not conj_mom:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    else:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot = hy.make_vars("x", "y", "z", "px", "py", "pz")

    H = get_cr3bp_hamiltonian(conj_mom=True)
    state_derivative = get_hamiltonian_state_derivative(H, conj_mom=conj_mom)
    state_derivative = convert_sympy_to_hy(state_derivative)

    if not batch_mode:
        return hy.taylor_adaptive(
            # Hamilton's equations.
            [
                (x, state_derivative[0]),
                (y, state_derivative[1]),
                (z, state_derivative[2]),
                (xdot, state_derivative[3]),
                (ydot, state_derivative[4]),
                (zdot, state_derivative[5]),
            ],
            # Initial conditions.
            ic,
        )
    else:
        return hy.taylor_adaptive_batch(
            # Hamilton's equations.
            [
                (x, state_derivative[0]),
                (y, state_derivative[1]),
                (z, state_derivative[2]),
                (xdot, state_derivative[3]),
                (ydot, state_derivative[4]),
                (zdot, state_derivative[5]),
            ],
            # Initial conditions.
            ic,
        )


def get_ta_var_raw(state, f, phi, dphidt, ic, ic_var, event=None):
    """
    Retrieves a variational Taylor-adaptive integrator using 'raw' inputs.

    Parameters
    ----------
    state: sp.matrix
        The symbols for the state
    f: sp.Matrix
        The symbolic state derivative expression
    phi: sp.Matrix
        The symbols for the variational equations
    dphidt: sp.Matrix
        The symbolic variational equations
    ic: np.ndarray
        The numerical state used to initialize the integrator
    ic_var: np.ndarray
        The numerical variational states used to initialize the integrator
    event: hy.core.t_event_dbl
        An event that can be used during the propagation

    Returns
    -------
    hy.taylor_adaptive
        A Taylor adaptive integrator
    """
    dyn = []
    for state_element, rhs in zip(state, f):
        dyn.append((state_element, rhs))
    for state_element, rhs in zip(phi.reshape(36, 1), dphidt.reshape(36, 1)):
        dyn.append((state_element, rhs))

    dyn_hy = convert_sympy_to_hy(dyn)
    # vsys = hy.var_ode_sys(dyn_hy, hy.var_args.vars, order=2)
    # assert(isinstance(vsys, hy.core.var_ode_sys))
    # assert(isinstance(ic+ic_var, list))
    # assert(isinstance((ic+ic_var)[0], float))
    if isinstance(event, hy.core.t_event_dbl):
        return hy.taylor_adaptive(
            # The ODEs.
            dyn_hy,
            # sys=vsys,
            # The initial conditions.
            state=ic + ic_var,
            # pars=[1.0]
            # Operate below machine precision
            # and in high-accuracy mode.
            t_events=[event],
            tol=1e-18,
            high_accuracy=True,
        )
    elif event is None:
        return hy.taylor_adaptive(
            # The ODEs.
            dyn_hy,
            # sys=vsys,
            # The initial conditions.
            state=ic + ic_var,
            # pars=[1.0]
            # Operate below machine precision
            # and in high-accuracy mode.
            tol=1e-18,
            high_accuracy=True,
        )
    else:
        raise RuntimeError("An error occurred. Review the event being passed.")


def get_ta_var(ic, ic_var, conj_mom=False, event=None):
    """
    Retrieves a Taylor-adaptive integrator.

    Parameters
    ----------
    ic: np.ndarray or list[np.ndarray]
        Initial state used for initializing the integrator
    ic_var: np.ndarray
        The numerical variational states used to initialize the integrator
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.
    event: hy.core.t_event_dbl
        An event that can be used during the propagation

    Returns
    -------
    hy.taylor_adaptive
        A Taylor adaptive integrator
    """
    phi = sp.Matrix([[sp.symbols(f"phi_{i}{j}") for j in range(6)] for i in range(6)])

    H = get_cr3bp_hamiltonian(conj_mom=True)
    f = sp.Matrix(6, 1, get_hamiltonian_state_derivative(H, conj_mom=conj_mom).tolist())
    if not conj_mom:
        x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz")
        state = sp.Matrix([x, y, z, vx, vy, vz])
    else:
        x, y, z, px, py, pz = sp.symbols("x y z px py pz")
        state = sp.Matrix([x, y, z, px, py, pz])

    dfdstate = f.jacobian(state)
    dphidt = dfdstate * phi

    return get_ta_var_raw(state, f, phi, dphidt, ic, ic_var, event)
