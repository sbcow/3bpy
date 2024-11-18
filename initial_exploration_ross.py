import numpy as np  # type: ignore
import sympy as sp
import heyoka as hy
import matplotlib.pyplot as plt
import copy


################
### Rotation ###
################


def convert_rotating_to_inertial_frame(vec, num_theta):
    theta = sp.symbols("theta")

    # Symbolic rotation matrix
    A = sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta), 0],
            [sp.sin(theta), sp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    # Check inverses == tranposes
    # sp.simplify(A.T * A)

    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, float)
    ):  # vec is 3x1 and theta is float
        return np.matmul(A.evalf(subs={theta: num_theta}), vec)
    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, np.ndarray)
    ):  # vec is 3x1 and theta is nx1 array
        return np.array([np.matmul(A.evalf(subs={theta: i}), vec) for i in num_theta])
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, float)
    ):  # vec is nx3 and theta is float
        return np.array(
            [np.matmul(A.evalf(subs={theta: num_theta}), vec_row) for vec_row in vec]
        )
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, np.ndarray)
    ):  # vec is nx3 and theta is nx1 array
        return np.array(
            [
                np.matmul(A.evalf(subs={theta: num_theta[it]}), vec_row)
                for it, vec_row in enumerate(vec)
            ]
        )
    else:
        raise RuntimeError(
            f"The provided vector: {vec} and theta: {num_theta} are malformed"
        )


def convert_inertial_to_rotating_frame(vec, num_theta):
    theta = sp.symbols("theta")

    # Symbolic rotation matrix
    A = sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta), 0],
            [sp.sin(theta), sp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    # Check inverses == tranposes
    # sp.simplify(A.T * A)

    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, float)
    ):  # vec is 3x1 and theta is float
        return np.matmul(A.T.evalf(subs={theta: num_theta}), vec)
    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, np.ndarray)
    ):  # vec is 3x1 and theta is nx1 array
        return np.array([np.matmul(A.T.evalf(subs={theta: i}), vec) for i in num_theta])
    elif (
        isinstance(vec, np.ndarray) and vec.shape[0] > 1 and vec.shape[1] == 3
    ):  # vec is nx3 and theta is float
        return np.array(
            [np.matmul(A.T.evalf(subs={theta: num_theta}), vec_row) for vec_row in vec]
        )
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, np.ndarray)
    ):  # vec is nx3 and theta is nx1 array
        return np.array(
            [
                np.matmul(A.T.evalf(subs={theta: num_theta[it]}), vec_row)
                for it, vec_row in enumerate(vec)
            ]
        )
    else:
        raise RuntimeError(
            f"The provided vector: {vec} and theta: {num_theta} are malformed"
        )


def convert_rotating_inertial_frame_test():

    # Single case
    vec = np.array([1, 0, 0], dtype=float)
    theta = np.pi / 2
    exp_vec = np.array([0, 1, 0], dtype=float)
    obt_vec = convert_rotating_to_inertial_frame(vec, theta)
    orig_obt_vec = convert_inertial_to_rotating_frame(obt_vec, theta)
    assert np.isclose(exp_vec, obt_vec.astype(float)).all()
    assert np.isclose(vec, orig_obt_vec.astype(float)).all()

    # Vec case
    n_of_points = 20
    theta_points = np.linspace(0, 2 * np.pi, n_of_points)
    theta = np.pi / 2
    coordinates = np.zeros((n_of_points, 3))
    # coordinates = np.repeat(np.array([1, 0, 0]), n_of_points)
    for i in range(n_of_points):
        current_theta_point = theta_points[i]
        coordinates[i, :] = np.array(
            [np.cos(current_theta_point), np.sin(current_theta_point), 0]
        )

    exp_vec = coordinates
    rot_obt_vec = convert_rotating_to_inertial_frame(coordinates, theta)
    obt_vec = convert_inertial_to_rotating_frame(rot_obt_vec, theta)
    assert np.isclose(exp_vec, obt_vec.astype(float)).all()


##########################
### Hamiltonian system ###
##########################


def convert_conj_mom_to_vel(state):
    vx = state[3] + state[1]
    vy = state[4] - state[0]
    vz = state[5]
    state[3:6] = np.array([vx, vy, vz])
    return state


def convert_vel_to_conj_mom(state):
    px = state[3] - state[1]
    py = state[4] + state[0]
    pz = state[5]
    state[3:6] = np.array([px, py, pz])
    return state


def get_u_bar(mu=None, position=None):

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
    E = get_cr3bp_hamiltonian(mu, state, conj_mom)
    return -2 * E


def get_current_symbols(expression):
    if isinstance(expression, sp.core.expr.Expr):
        symbol_dict = {it: symb for it, symb in enumerate(expression.free_symbols)}


def get_hamiltonian_state_derivative(H, mu_val=None, state=None, conj_mom=False):

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


def get_ta(H, init_cond, batch_mode=False, conj_mom=False):
    if not conj_mom:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    else:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot = hy.make_vars("x", "y", "z", "px", "py", "pz")

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
            init_cond,
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
            init_cond,
        )


def get_ta_var_raw(state, f, phi, dphidt, ic, ic_var, event=None):
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

def get_cr3bp_jacobian(mu_val=None, state=None, conj_mom=False):
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


def get_po_ig(xL, mu_val, use_real=True, disp=1e-4, direction=True, conj_mom=False):
    state = np.array([xL, 0, 0, 0, 0, 0])
    jacobian = get_cr3bp_jacobian(mu_val, state, conj_mom)
    eigs = np.linalg.eig(jacobian)
    first_oscillatory_index = next(
        i for i, x in enumerate(eigs.eigenvalues) if np.iscomplex(x) if np.imag(x) < 0
    )
    # Get eigen direction for oscillatory motion
    if use_real:
        first_oscillatory_direction = np.real(
            eigs.eigenvectors.T[first_oscillatory_index]
        )
    else:
        first_oscillatory_direction = np.imag(
            eigs.eigenvectors.T[first_oscillatory_index]
        )

    # Period guess
    po_period_guess = abs(
        2 * np.pi / np.imag(eigs.eigenvalues[first_oscillatory_index])
    )
    po_ic = (state + first_oscillatory_direction * disp).tolist() if direction is True else (state - first_oscillatory_direction * disp).tolist()

    return po_ic, po_period_guess


def plot_zero_vel_curves(ax, mu_val, ic, xL, buffer=1.5, resolution=2000):
    # Setup zero-velocity curves
    E = get_cr3bp_hamiltonian(mu=mu_val, state=ic, conj_mom=False)

    # Find the forbidden region
    xx = np.linspace(xL - buffer, xL + buffer, resolution)
    yy = np.linspace(-buffer, buffer, resolution)
    x_grid, y_grid = np.meshgrid(xx, yy)
    symbs = ["x", "y", "z", "xdot", "ydot", "zdot"]
    potentials = get_u_bar(mu_val, (x_grid, y_grid, np.zeros(np.shape(x_grid))))

    # zero velocity curve
    ax.imshow(
        (potentials >= E).astype(int),
        extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
        origin="lower",
        cmap="Greens",
    )


def corrector(ta, x0):

    period = ta.time
    Phi_T = ta.state[6:].reshape((6, 6))
    dx_T = ta.state[:6] - x0
    # print("error was:", np.linalg.norm(dx_T))
    dx0 = -np.linalg.inv(Phi_T - np.eye(6)) @ dx_T
    x0 += dx0
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(period)
    # print("new error is:", np.linalg.norm(ta.state[:6] - x0))
    return ta, x0, np.linalg.norm(ta.state[:6] - x0)


def corrector_phasing(ta, x0, conj_mom=False):

    x0 = np.array(x0)
    period = ta.time
    Phi_T = ta.state[6:].reshape((6, 6))
    dx_T = ta.state[:6] - x0
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f = get_hamiltonian_state_derivative(
        H, mu_val=ta.pars[0], state=ta.state[:6], conj_mom=conj_mom
    ).reshape(-1, 1)
    # print("error was:", np.linalg.norm(dx_T))
    dx_T = np.insert(dx_T, -1, 0).reshape(
        -1, 1
    )  # Add zero corresponding to phasing condition
    A = np.concatenate((Phi_T - np.eye(6), f), axis=1)
    # We add the Poincare phasing condition as a last equation
    phasing_cond = np.insert(f, -1, 0).reshape((1, -1))
    A = np.concatenate((A, phasing_cond))  # Add extra row 7x1
    delta = -np.linalg.inv(A) @ dx_T.reshape(
        7,
    )
    x0 += delta[:6]
    period = period + delta[-1]
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(period)
    # print("new error is:", np.linalg.norm(ta.state[:6] - x0))
    return ta, x0, np.linalg.norm(ta.state[:6] - x0)


# def corrector_pa(ta, x0, t0, dx, taux, taut, ds):

#     curr_dx = copy.deepcopy(dx[:6])
#     curr_dT = copy.deepcopy(dx[6])

#     curr_x = copy.copy(x0)
#     curr_T = copy.copy(t0)

#     # 1 - We compute the state transition matrix Phi (integrating the full EOM from curr_x for t_final)
#     ta.time = 0.0
#     ta.state[:] = curr_x.tolist() + np.eye(6).reshape((36,)).tolist()
#     _ = ta.propagate_until(curr_T)
#     Phi = ta.state[6:].reshape((6, 6))
#     # 2 - We compute the dynamics f (at the initial state) - just for poincare phasing error
#     # f_dyn0 = cf_f(x0, pars = [mu]).reshape(-1,1)
#     H = get_cr3bp_hamiltonian()
#     f_dyn0 = get_hamiltonian_state_derivative(H, ta.pars[0], x0 - dx[:6]).reshape(-1, 1)
#     # 3 - We compute the dynamics f (at curr_T)
#     # f_dynT = cf_f(ta.state[:6], pars = [mu]).reshape(-1,1)
#     f_dynT = get_hamiltonian_state_derivative(H, ta.pars[0], ta.state[:6]).reshape(-1, 1)
#     # 4 - Assemble the function (the full nonlinear system) value at the current point
#     state_err = ta.state[:6] - curr_x
#     # Error on Poincare phasing
#     poin_err = curr_dx @ f_dyn0
#     # Error on pseudo arc-length
#     pseudo_err = taux @ curr_dx + taut * curr_dT - ds
#     b = np.zeros((8, 1))
#     b[:6, 0] = -state_err
#     b[6, 0] = -poin_err
#     b[7, 0] = -pseudo_err
#     # print(np.linalg.norm(state_err), poin_err, pseudo_err)
#     toterror = (
#         np.abs(np.linalg.norm(state_err)) + np.abs(poin_err) + np.abs(pseudo_err)
#     )
#     # 5 - Assemble the matrix A (gradient)
#     A = np.concatenate((Phi - np.eye(6), f_dynT.T))
#     A = np.concatenate((A, np.insert(f_dynT, -1, 0).reshape((-1, 1))), axis=1)
#     # add the tau row
#     tmp = np.insert(taux, 6, taut).reshape(1, -1)
#     A = np.concatenate((A, tmp))

#     # Solve for new y = [dx, dT]
#     y = (np.linalg.inv(A.T @ A) @ A.T @ b).reshape(
#         -1,
#     )
#     curr_dx += y[:6]
#     curr_dT += y[6]
#     curr_x = x0 - dx[:6] + curr_dx.T
#     curr_T = t0 - dx[6] + curr_dT
#     return np.insert(curr_dx.T, 6, curr_dT)
#     return ta, x0, np.linalg.norm(ta.state[:6] - x0)


def diff_corr(ta, x0, tol=1e-10, max_iter=100):
    err = 1
    it = 0
    while err > tol:
        ta, x0, err = corrector(ta, x0)
        it += 1
        if it > max_iter:
            raise RuntimeError(f"Correction didn't converge. Error is still: {err}")
    return ta, x0, err

def diff_corr_phasing(ta, x0, tol=1e-10, max_iter=100, conj_mom=False):
    err = 1
    it = 0
    while err > tol:
        ta, x0, err = corrector_phasing(ta, x0, conj_mom=conj_mom)
        it += 1
        if it > max_iter:
            raise RuntimeError(f"Correction didn't converge. Error is still: {err}")
    return ta, x0, err

def diff_corr_pa(ta, x0, t0, ds, dx, taux, taut, tol=1e-12, max_iter=100):
    """
    Here, ta is the TA integrator, x0 is the periodic orbit (closed solution)
    that is to be expanded, t0 is the period of that closed orbit, dx is the
    difference in state as predicted by the predictor_pseudoarc(). taux and taut are
    the tangential vector as given by the predictor_pseudoarc(). ds is the variation.
    """
    flag_tol = False

    curr_dx = copy.deepcopy(dx[:6])
    curr_dT = copy.deepcopy(dx[6])

    # Linearization point
    curr_x = x0 + curr_dx.T
    curr_T = t0 + curr_dT

    for i in range(max_iter):
        # 1 - We compute the state transition matrix Phi (integrating the full EOM from curr_x for t_final)
        ta.time = 0.0
        ta.state[:] = curr_x.tolist() + np.eye(6).reshape((36,)).tolist()
        _ = ta.propagate_until(curr_T)
        Phi = ta.state[6:].reshape((6, 6))
        # 2 - We compute the dynamics f (at the initial state)
        # f_dyn0 = cf_f(x0, pars = [mu]).reshape(-1,1)
        H = get_cr3bp_hamiltonian()
        f_dyn0 = get_hamiltonian_state_derivative(H, ta.pars[0], x0).reshape(-1, 1)
        # 3 - We compute the dynamics f (at curr_T)
        # f_dynT = cf_f(ta.state[:6], pars = [mu]).reshape(-1,1)
        f_dynT = get_hamiltonian_state_derivative(H, ta.pars[0], ta.state[:6]).reshape(-1, 1)
        # 4 - Assemble the function (the full nonlinear system) value at the current point
        state_err = ta.state[:6] - curr_x
        # Error on Poincare phasing
        poin_err = curr_dx @ f_dyn0
        # Error on pseudo arc-length
        pseudo_err = taux @ curr_dx + taut * curr_dT - ds
        b = np.zeros((8, 1))
        b[:6, 0] = -state_err
        b[6, 0] = -poin_err
        b[7, 0] = -pseudo_err
        # print(np.linalg.norm(state_err), poin_err, pseudo_err)
        toterror = (
            np.abs(np.linalg.norm(state_err)) + np.abs(poin_err) + np.abs(pseudo_err)
        )
        if toterror < tol:
            flag_tol = True
            break
        # 5 - Assemble the matrix A (gradient)
        A = np.concatenate((Phi - np.eye(6), f_dynT.T))
        A = np.concatenate((A, np.insert(f_dynT, -1, 0).reshape((-1, 1))), axis=1)
        # add the tau row
        tmp = np.insert(taux, 6, taut).reshape(1, -1)
        A = np.concatenate((A, tmp))

        # Solve for new y = [dx, dT]
        y = (np.linalg.inv(A.T @ A) @ A.T @ b).reshape(
            -1,
        )
        curr_dx += y[:6]
        curr_dT += y[6]
        curr_x = x0 + curr_dx.T
        curr_T = t0 + curr_dT
    if flag_tol:
        print("Exit condition - Accuracy")
    else:
        print("Exit condition - Maximum Iterations")
    print("Accuracy achieved: ", toterror)
    print("Iterations: ", i)

    # ta.time = 0.0
    # ta.state[:] = curr_x.tolist() + np.eye(6).reshape((36,)).tolist()
    # _ = ta.propagate_until(curr_T)
    # print("new error is:", np.linalg.norm(ta.state[:6] - x0))
    # return np.insert(curr_dx.T, 6, curr_dT)
    return ta, curr_dx, curr_dT, toterror

def predictor_pseudoarc(ta, x0, t0, ds):
    # 1 - We compute the state transition matrix Phi (integrating the full EOM for t_final)
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(t0)
    Phi = ta.state[6:].reshape((6, 6))
    # 2 - We compute the dynamics f (at zero, but for periodic orbits this is the same at T)
    H = get_cr3bp_hamiltonian()
    f_dyn = get_hamiltonian_state_derivative(H, ta.pars[0], x0).reshape(-1, 1)
    # 3 - Assemble the matrix A
    A = np.concatenate((Phi - np.eye(6), f_dyn.T))
    A = np.concatenate((A, np.insert(f_dyn, -1, 0).reshape((-1, 1))), axis=1)
    # 4 - Compute the tangent vector tau
    tauT = 1
    taux = -np.linalg.inv(A[:, :6].T @ A[:, :6]) @ (A[:, :6].T @ A[:, -1]) * tauT
    norm = np.sqrt(taux @ taux + tauT * tauT)
    tauT /= norm
    taux /= norm
    # 5 - Add to the matrix A
    tmp = np.insert(taux, 6, tauT).reshape(1, -1)
    A = np.concatenate((A, tmp))
    # 6 - Compute the b vector
    b = np.zeros((8, 1))
    b[7, 0] = ds
    # 6 - Predict an initial guess
    dx = np.linalg.inv(A.T @ A) @ (A.T @ b).reshape((-1))
    return dx, taux, tauT


def predictor(ta, cont_param=0, variation=1e-4):
    Phi = ta.state[6:].reshape((6, 6))
    state_T = ta.state[:6]
    # Compute the dynamics from its expressions
    H = get_cr3bp_hamiltonian()
    f_dyn = get_hamiltonian_state_derivative(H, ta.pars[0], state_T).reshape(-1, 1)
    # Computing the full A
    A = np.concatenate((Phi - np.eye(6), f_dyn.T))
    fullA = np.concatenate((A, np.insert(f_dyn, -1, 0).reshape((-1, 1))), axis=1)
    # Computing the A resulting from fixing the continuation parameter to a selected state.
    A = fullA[:, list(set(range(7)) - set([cont_param]))]
    # Now we multiply the variation of the continuation parameter with the
    # corresponding STM derivatives to get the distance of the final state given
    # the initial variation of the contination parameter
    b = -fullA[:, [cont_param]] * variation
    # We solve.
    dx = np.linalg.inv((A.T @ A)) @ (A.T @ b)
    # Assembling back the full state (x,y,z,px,py,pz,T)
    dx = np.insert(dx, cont_param, variation)
    return dx


def perform_predictor_corrector_step(
    ta_var,
    cont_type='basic',
    predictor_func_args = (),
    corrector_func_args = (),
    tol=1e-10,
    direction=True,
    max_iter=100
):
    
    taux = None
    taut = None
    # Perform predictor step
    if cont_type == 'basic' or cont_type == 'phasing':
        dx = predictor(ta_var, *predictor_func_args)
    elif cont_type == 'pseudoarc':
        dx, taux, taut = predictor_pseudoarc(ta_var, *predictor_func_args)
    else:
        raise ValueError(f"{cont_type} currently not recognized/implemented")

    # Move ic in one of two directions
    if direction:
        po_ic_cont = (np.array(ta_var.state[:6]) + dx[:6]).tolist()
        new_T = ta_var.time + dx[-1]
    else:
        po_ic_cont = (np.array(ta_var.state[:6]) - dx[:6]).tolist()
        new_T = ta_var.time - dx[-1]

    # Propagate predicted step
    timesteps = np.linspace(0, new_T, 2000)
    ta_var.time = 0.0
    ta_var.state[:] = po_ic_cont + np.eye(6).reshape((36,)).tolist()
    _ = ta_var.propagate_grid(timesteps)
    
    # Perform correction
    if cont_type == 'basic':
        return diff_corr(ta_var, po_ic_cont, tol=tol, max_iter=max_iter)
    elif cont_type == 'phasing':
        return diff_corr_phasing(ta_var, po_ic_cont, tol=tol, max_iter=max_iter)
    elif cont_type == 'pseudoarc':
        corrector_func_args = (po_ic_cont, new_T, corrector_func_args, dx, taux, taut)
        return diff_corr_pa(ta_var, *corrector_func_args, tol=tol, max_iter=max_iter)
    # return corrector_func(ta_var, corrector_func_args, tol=tol, max_iter=max_iter)
    # try:
    #     ta_var, x0 = corrector_func(ta_var, po_ic_cont, with_phasing=with_phasing, tol=tol)
    # except RuntimeError:
    #     print(f"variation decreased from {variation} to {variation/1.1}")
    #     variation /= 1.1
    #     ta_var, x0 = perform_predictor_corrector_step(
    #         ta_var,
    #         cont_param=0,
    #         predictor_func=predictor,
    #         corrector_func=diff_corr,
    #         with_phasing=True,
    #         tol=1e-10,
    #         variation=variation
    #     )

    # return ta_var, x0


def get_mu_bar(mu, xL):
    return mu / np.abs(xL - 1 + mu) ** 3 + (1 - mu) / np.abs(xL + mu) ** 3


# def get_Phi_at_L(mu):
#     if mu is None:
#         x, y, z, mu = sp.symbols("x y z par[0]")
#         u_bar = get_u_bar(mu)
#         sp.diff(u_bar, x)
#     else:
#         x, y, z = sp.symbols("x y z")
#         u_bar = get_u_bar(mu)
#         vars = sp.Matrix([x, y, z])


def get_eigen_values_at_L(mu, xL):
    mubar = get_mu_bar(mu, xL)
    lam = np.sqrt(0.5 * (mubar - 2 + np.sqrt(9 * mubar**2 - 8 * mubar)))
    nu = np.sqrt(-0.5 * (mubar - 2 - np.sqrt(9 * mubar**2 - 8 * mubar)))
    return lam, nu


def get_sigma_tau_constants(mu, xL):
    mubar = get_mu_bar(mu, xL)
    lam, nu = get_eigen_values_at_L(mu, xL)
    a = 1 + 2 * mubar
    b = mubar - 1

    sigma = 2 * lam / (lam**2 + b)
    tau = -(nu**2 + a) / (2 * nu)

    return sigma, tau


def get_eigenbasis_vectors(mu, xL):
    lam, nu = get_eigen_values_at_L(mu, xL)
    sigma, tau = get_sigma_tau_constants(mu, xL)

    u1 = np.array([1, -sigma, lam, -lam * sigma])
    u2 = np.array([1, sigma, -lam, -lam * sigma])
    u = np.array([1, 0, 0, nu * tau])
    v = np.array([0, tau, nu, 0])
    return u1, u2, u, v


# def find_lagrange_points_old(mu):
#     """
#     This function does not work properly, specifically the 2nd L point is incorrect. The 3rd one is also an approximation.
#     """

#     # Find L point
#     pL1 = np.polynomial.Polynomial([1, -1 * (3 - mu), (3 - 2 * mu), -mu, 2 * mu, -mu])
#     rootsL1 = np.roots(pL1.coef)
#     pL2 = np.polynomial.Polynomial([1, -1 * (3 - mu), (3 - 2 * mu), -mu, -2 * mu, -mu])
#     rootsL2 = np.roots(pL2.coef)
#     for it in range(len(rootsL1)):
#         if np.isreal(rootsL1[it]):
#             gammaL1 = rootsL1[it].real
#         if np.isreal(rootsL2[it]):
#             gammaL2 = rootsL2[it].real
#     xL1 = (1 - mu) - gammaL1
#     xL2 = (1 - mu) + gammaL2
#     xL3 = -mu - (1 - 7 * mu / 12)  # Approximation wikipedia

#     # By convention, L4 and L5 lie at:
#     xL4, yL4 = (0.5 - mu, np.sqrt(3) / 2)
#     xL5, yL5 = (0.5 - mu, -np.sqrt(3) / 2)

#     return xL1, xL2, xL3, (xL4, yL4), (xL5, yL5)


def find_lagrange_points(mu_val):
    x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f = sp.Matrix(6, 1, get_hamiltonian_state_derivative(H, conj_mom=False).tolist())
    lp = sp.solve(f[3].evalf(subs={mu: mu_val, vy: 0, y: 0, z: 0}), x)
    lp = [float(x) for x in lp]
    lp.append(tuple((0.5 - mu_val, np.sqrt(3) / 2)))
    lp.append((0.5 - mu_val, -np.sqrt(3) / 2))
    return lp


###############################
### Differential correction ###
###############################


# def get_variational_symbols():
#     symbols_phi = []
#     for i in range(6):
#         for j in range(6):
#             # Here we define the symbol for the variations
#             symbols_phi.append("phi_" + str(i) + str(j))
#     phi = np.array(hy.make_vars(*symbols_phi)).reshape((6, 6))

#     dfdx = []

#     H = get_cr3bp_hamiltonian(conj_mom=False)
#     f = get_hamiltonian_state_derivative(H, conj_mom=False)
#     x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz")
#     state = sp.Array[x, y, z, vx, vy, vz]

#     dfdstate = f.diff(state)
#     for i in range(6):
#         for j in range(6):
#             dfdx.append(sp.diff(f[i], x[j]))
#     dfdx = np.array(dfdx).reshape((6, 6))

#     dphidt = dfdx @ phi

# import numpy as np

# def prtbp(t, x):
#     global mu

#     mu1 = 1 - mu  # mass of larger primary (nearest origin on left)
#     mu2 = mu      # mass of smaller primary (furthest from origin on right)

#     r3 = ((x[0] + mu2) ** 2 + x[1] ** 2) ** 1.5  # r: distance to m1, LARGER MASS
#     R3 = ((x[0] - mu1) ** 2 + x[1] ** 2) ** 1.5  # R: distance to m2, smaller mass

#     xdot = np.zeros(4)
#     xdot[0] = x[2]
#     xdot[1] = x[3]
#     xdot[2] = x[0] - (mu1 * (x[0] + mu2) / r3) - (mu2 * (x[0] - mu1) / R3) + 2 * x[3]
#     xdot[3] = x[1] - (mu1 * x[1] / r3) - (mu2 * x[1] / R3) - 2 * x[2]

#     return xdot

# import numpy as np
# from scipy.integrate import solve_ivp

# def PHIget(x0, tf):
#     global tol
#     OPTIONS = {'rtol': 3 * tol, 'atol': tol}

#     N = len(x0)

#     PHI_0 = np.zeros(N**2 + N)
#     PHI_0[:N**2] = np.eye(N).flatten()
#     PHI_0[N**2:N**2 + N] = x0

#     if N == 4:
#         sol = solve_ivp(var2D, [0, tf], PHI_0, method='RK45', **OPTIONS)
#     elif N == 6:
#         sol = solve_ivp(var3D, [0, tf], PHI_0, method='RK45', **OPTIONS)

#     t = sol.t
#     PHI = sol.y.T
#     x = PHI[:, N**2:N**2 + N]  # trajectory
#     phi_T = PHI[-1, :N**2].reshape(N, N)  # monodromy matrix, PHI(0,T)

#     return x, t, phi_T, PHI

# def var2D(t, PHI):
#     global FORWARD, mu

#     mu1 = 1 - mu
#     mu2 = mu

#     x = PHI[16:20]
#     phi = PHI[:16].reshape(4, 4)

#     r2 = (x[0] + mu) ** 2 + x[1] ** 2  # r: distance to m1, LARGER MASS
#     R2 = (x[0] - mu1) ** 2 + x[1] ** 2  # R: distance to m2, smaller mass
#     r3 = r2 ** 1.5
#     r5 = r2 ** 2.5
#     R3 = R2 ** 1.5
#     R5 = R2 ** 2.5

#     omgxx = 1 + (mu1 / r5) * (3 * (x[0] + mu2) ** 2) + (mu2 / R5) * (3 * (x[0] - mu1) ** 2) - (mu1 / r3 + mu2 / R3)
#     omgyy = 1 + (mu1 / r5) * (3 * x[1] ** 2) + (mu2 / R5) * (3 * x[1] ** 2) - (mu1 / r3 + mu2 / R3)
#     omgxy = 3 * x[1] * (mu1 * (x[0] + mu2) / r5 + mu2 * (x[0] - mu1) / R5)

#     F = np.array([[0, 0, 1, 0],
#                   [0, 0, 0, 1],
#                   [omgxx, omgxy, 0, 2],
#                   [omgxy, omgyy, -2, 0]])

#     phidot = F @ phi  # variational equation

#     PHIdot = np.zeros(20)
#     PHIdot[:16] = phidot.flatten()
#     PHIdot[16] = x[2]
#     PHIdot[17] = x[3]
#     PHIdot[18] = x[0] - (mu1 * (x[0] + mu2) / r3) - (mu2 * (x[0] - mu1) / R3) + 2 * x[3]
#     PHIdot[19] = x[1] - (mu1 * x[1] / r3) - (mu2 * x[1] / R3) - 2 * x[2]
#     PHIdot[16:20] *= FORWARD

#     return PHIdot


# import numpy as np
# from scipy.integrate import solve_ivp

# def haloy(t1):
#     global t0_z, x0_z, x1_zgl

#     if t1 == t0_z:
#         x1_zgl = x0_z
#     else:
#         xx, tt = intDT(x0_z, t0_z, t1)
#         x1_zgl = xx[-1, :]

#     return x1_zgl[1]

# def intDT(x0, t0, tf):
#     global FORWARD, tol

#     options = {'rtol': 3 * tol, 'atol': tol}

#     m = len(x0)

#     if m == 4:
#         sol = solve_ivp(prtbp, [t0, tf], x0, **options)
#         t, x = sol.t, sol.y.T
#     elif m == 6:
#         sol = solve_ivp(rtbp, [t0, tf], x0, **options)
#         t, x = sol.t, sol.y.T

#     t = FORWARD * t  # for backwards integration, time is like a countdown

#     return x, t

# import numpy as np
# from scipy.optimize import fsolve

# def find0(x0):
#     global t0_z, x0_z, x1_zgl

#     tolzero = 1.e-10
#     options = {'xtol': tolzero, 'maxfev': 1000}

#     t0_z = np.pi / 2 - 0.15
#     xx = int(x0)
#     x0_z = xx[-1, :]
#     t1_z = fsolve(haloy, t0_z, **options)[0]
#     x1_z = x1_zgl

#     del globals()['t0_z']
#     del globals()['x0_z']
#     del globals()['x1_zgl']

#     return t1_z, x1_z


# def liadifcor(x0g, xL, mu, show=0):
# # [x0,t1]=liadifcor(x0g,show)
# #
# # This is the differential correction routine to create planar periodic
# # Liapunov orbits about L1,L2, or L3 in the CR3BP. It keeps the initial
# # x value constant and varies the y-velocity value.
# #
# # output: x0  = initial state on the Liapunov (on the xz-plane) in nondim.
# # 	CR3BP coords.
# #   t1  = half-period of Liapunov orbit in nondim. CR3BP time
# #
# # input: x0g  = first guess of initial state on the Liapunov orbit
# #  show = 1 to plot successive orbit  (default=0)
# #
# # ------------------------------------------------------------------------
# # CR3BP with the LARGER MASS, m1 to the left of the origin at (-mu,0)
# # and m2, or the planet (ie. Earth), is at (1 - mu, 0)
# #
# #                L4
# # -L3----m1--+-------L1--m2--L2-
# #                L5
# #
# # Shane Ross (revised 8.28.97)

#     mubar = mu / np.abs(xL - 1 + mu)**3 + (1 - mu) / np.abs(xL + mu)**3

#     a = 1 + 2 * mubar
#     b = mubar - 1

#     t0 = 0
#     dxdot1 = 1
#     MAXdxdot1 = 1.e-12  # measure of precision of perpendicular x-axis crossing
#     attempt = 0
#     MAXattempt = 25

#     while abs(dxdot1) > MAXdxdot1:
#         if attempt > MAXattempt:
#             ERROR = 'OCCURED'
#             break

#         t1, xx1 = find0(x0g)  # find x-axis crossing
#         attempt += 1

#         x1, y1 = xx1[0], xx1[1]
#         dxdot1, ydot1 = xx1[2], xx1[3]

#         x, t, phi_t1, PHI = PHIget(x0g, t1)

#         if show == 1:
#             plt.scatter(x, 0)
#             plt.plot(a * x[0, 0], a * x[0, 1], 'wo')
#             plt.axis([min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1])])
#             plt.axis('equal')

#         mu2 = 1 - mu
#         r1 = ((x1 + mu) ** 2 + y1 ** 2)
#         r2 = ((x1 - mu2) ** 2 + y1 ** 2)
#         rho1 = 1 / r1 ** 1.5
#         rho2 = 1 / r2 ** 1.5

#         omgx1 = -(mu2 * (x1 + mu) * rho1) - (mu * (x1 - mu2) * rho2) + x1

#         xdotdot1 = (2 * ydot1 + omgx1)

#         C1 = phi_t1[2, 3]
#         C2 = phi_t1[1, 3]
#         C3 = (C1 - (1 / ydot1) * xdotdot1 * C2)
#         C4 = np.linalg.inv(C3) * (-dxdot1)
#         dydot0 = C4

#         x0g[3] = x0g[3] + dydot0

#     x0 = x0g
#     return x0, t1


# ##############
# ### Script ###
# ##############

# convert_rotating_inertial_frame_test()

# ### Variables

# mu = 0.01

# xL1, xL2, xL3, (xL4, yL4), (xL5, yL5) = find_lagrange_points(mu)
# u1, u2, u, v = get_eigenbasis_vectors(xL1, mu)

# # displacements = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# # displacements = np.linspace(-1e-2, 1e-2, 20)
# displacements = [-0.2, 0.0, 0.2, 0.4]
# init_cond = np.array([-0.72, 0.0, 0.0, 0.0, -0.2, 0.0])
# form_init_cond = np.vstack([init_cond + disp * np.array([1, 0, 0, 0, 0, 0]) for disp in displacements]).T
# # displacements = [1e-5, 1e-4]
# # form_init_cond = np.repeat(init_cond, len(displacements)).reshape(len(init_cond), -1)
# # for it, disp in enumerate(displacements):
# #     form_init_cond[[0, 1, 3, 4], it] += disp * u1
# # form_init_cond = init_cond.T.tolist()

# H = get_cr3bp_hamiltonian(mu, conj_mom=False)
# ta = get_ta(H, form_init_cond.tolist(), batch_mode=True)
# ta.set_time(0.0)
# nsteps = 2000
# epochs = np.repeat(np.linspace(0, np.pi / 2, nsteps), len(displacements)).reshape(
#     nsteps, -1
# )

# E = [get_cr3bp_hamiltonian(mu, state=ic, conj_mom=False) for ic in form_init_cond.T]

# # 1. the outcome of the integration,
# # 2. the minimum and maximum integration timesteps
# # 3. that were used in the propagation,
# # 4. the total number of steps that were # taken.
# # 5. The fifth value returned by propagate_grid() is the step callback, if provided
# # by the user (see below). Otherwise, None is returned.
# # 6. The sixth value returned by propagate_grid() is a 2D array containing the
# # state of the system at the time points in the grid:
# _, out = ta.propagate_grid(epochs)

# # def check_energy_consistency(states):
#     # for i in states

# # check_energy_consistency(out[5])

# m1_pos = np.array([-mu, 0, 0])
# m2_pos = np.array([1 - mu, 0, 0])
# L1_pos = np.array([xL1, 0, 0])
# L2_pos = np.array([xL2, 0, 0])
# L3_pos = np.array([xL3, 0, 0])
# L4_pos = np.array([xL4, yL4, 0])
# L5_pos = np.array([xL5, yL5, 0])


# m1_in_pos = convert_rotating_to_inertial_frame(m1_pos, epochs[:, 0])
# m2_in_pos = convert_rotating_to_inertial_frame(m2_pos, epochs[:, 0])
# L1_in_pos = convert_rotating_to_inertial_frame(L1_pos, epochs[:, 0])
# L2_in_pos = convert_rotating_to_inertial_frame(L2_pos, epochs[:, 0])
# L3_in_pos = convert_rotating_to_inertial_frame(L3_pos, epochs[:, 0])
# L4_in_pos = convert_rotating_to_inertial_frame(L4_pos, epochs[:, 0])
# L5_in_pos = convert_rotating_to_inertial_frame(L5_pos, epochs[:, 0])

# # Find the forbidden region
# xx = np.linspace(-1.5, 1.5, 2000)
# yy = np.linspace(-1.5, 1.5, 2000)
# x_grid, y_grid = np.meshgrid(xx, yy)
# symbs = ["x", "y", "z", "xdot", "ydot", "zdot"]
# potentials = get_u_bar(mu, (x_grid, y_grid, np.zeros(np.shape(x_grid))))

# ################
# ### Plotting ###
# ################

# fig = plt.figure(figsize=(5, 10))
# fig.add_subplot(2, 1, 1)


# # s/c trajectory
# for i in range(len(out[0, 0, :])):
#     plt.plot(out[:, 0, i], out[:, 1, i], c="b", linestyle="--", linewidth=1)
#     plt.scatter(out[-1, 0, i], out[-1, 1, i], c="b", s=5)

# # masses
# plt.scatter(-mu, 0, c="r", s=20)  # m1
# plt.scatter(1 - mu, 0, c="r", s=20)  # m2

# # Lagrange points
# plt.scatter(xL1, 0, c="k", s=10)  # m2
# plt.scatter(xL2, 0, c="k", s=10)  # m2
# plt.scatter(xL3, 0, c="k", s=10)  # m2
# plt.scatter(xL4, yL4, c="k", s=10)  # m2
# plt.scatter(xL5, yL5, c="k", s=10)  # m2

# # zero velocity curve
# plt.imshow((potentials >= E[1]),
#     extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
#     origin="lower",
#     cmap="Greens"
#     )

# plt.title(f"Top-down view - mu: {mu} - Rotating frame")
# plt.xlabel("x [AU]")
# plt.ylabel("y [AU]")
# max_bounds = 1.2
# # plt.ylim([np.min(out[:, 1]) if np.min(out[:, 1]) < -1.2 else -1.2, np.max(out[:, 1]) if np.max(out[:, 1])> 1.2 else 1.2])
# # plt.xlim([np.min(out[:, 0]) if np.min(out[:, 0]) < -1.2 else -1.2, np.max(out[:, 0]) if np.max(out[:, 0])> 1.2 else 1.2])
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# # plt.grid()
# plt.tight_layout()

# fig.add_subplot(2, 1, 2)
# # s/c trajectory
# converted_coords = []
# for i in range(len(out[0, 0, :])):
#     converted_coords.append(
#         convert_rotating_to_inertial_frame(out[:, 0:3, i], epochs[:, 0])
#     )
#     plt.plot(
#         converted_coords[i][:, 0],
#         converted_coords[i][:, 1],
#         c="b",
#         linestyle="--",
#         linewidth=1,
#     )
#     plt.scatter(converted_coords[i][-1, 0], converted_coords[i][-1, 1], c="b", s=5)

# # masses
# plt.plot(m1_in_pos[:, 0], m1_in_pos[:, 1], "ro", markersize=1)  # m1
# plt.scatter(m1_in_pos[-1, 0], m1_in_pos[-1, 1], c="r", s=20)  # m1
# plt.plot(m2_in_pos[:, 0], m2_in_pos[:, 1], "ro", markersize=1)  # m2
# plt.scatter(m2_in_pos[-1, 0], m2_in_pos[-1, 1], c="r", s=20)  # m2

# # Lagrange points
# plt.plot(L1_in_pos[:, 0], L1_in_pos[:, 1], "ko", markersize=1)  # m2
# plt.scatter(L1_in_pos[-1, 0], L1_in_pos[-1, 1], c="k", s=10)  # m2
# plt.plot(L2_in_pos[:, 0], L2_in_pos[:, 1], "ko", markersize=1)  # m2
# plt.scatter(L2_in_pos[-1, 0], L2_in_pos[-1, 1], c="k", s=10)  # m2
# plt.plot(L3_in_pos[:, 0], L3_in_pos[:, 1], "ko", markersize=1)  # m2
# plt.scatter(L3_in_pos[-1, 0], L3_in_pos[-1, 1], c="k", s=10)  # m2
# plt.plot(L4_in_pos[:, 0], L4_in_pos[:, 1], "ko", markersize=1)  # m2
# plt.scatter(L4_in_pos[-1, 0], L4_in_pos[-1, 1], c="k", s=10)  # m2
# plt.plot(L5_in_pos[:, 0], L5_in_pos[:, 1], "ko", markersize=1)  # m2
# plt.scatter(L5_in_pos[-1, 0], L5_in_pos[-1, 1], c="k", s=10)  # m2

# plt.title(f"Top-down view - mu: {mu} - Inertial frame")
# plt.xlabel("X [AU]")
# plt.ylabel("Y [AU]")
# max_bounds = 1.2
# # plt.ylim([np.min(converted_coords[:, 1]) if np.min(converted_coords[:, 1]) < -1.2 else -1.2, np.max(converted_coords[:, 1]) if np.max(converted_coords[:, 1])> 1.2 else 1.2])
# # plt.xlim([np.min(converted_coords[:, 0]) if np.min(converted_coords[:, 0]) < -1.2 else -1.2, np.max(converted_coords[:, 0]) if np.max(converted_coords[:, 0])> 1.2 else 1.2])
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# # plt.grid()
# plt.tight_layout()

# plt.show()


# # zero velocity curve
# plt.figure(figsize=(8, 8))
# for i in range(len(E)):
#     plt.subplot(2, 2, (i+1))
#     plt.imshow((potentials >= E[i]),
#         extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
#         origin="lower",
#         cmap="Greens"
#         )
# plt.show()

# def get_u_bar(mu=None, position=None):

#     if position is None and mu is None:
#         x, y, z, mu = sp.symbols("x y z par[0]")
#         mu2 = -mu
#         mu1 = 1-mu
#         r1 = sp.sqrt((x + mu2) ** 2 + y**2 + z**2)
#         r2 = sp.sqrt((x - mu1) ** 2 + y**2 + z**2)

#         U = (-mu1 / r1) - (mu2 / r2) - (0.5 * mu1 * mu2)
#         return -0.5 * (x**2 + y**2) + U

#     elif position is not None and mu is not None:
#         mu2 = -mu
#         mu1 = 1 - mu
#         x, y, z = position
#         r1 = np.sqrt((x + mu2) ** 2 + y**2 + z**2)
#         r2 = np.sqrt((x - mu1) ** 2 + y**2 + z**2)

#         U = -mu1 / r1 - mu2 / r2 - 0.5 * mu1 * mu2
#         return -0.5 * (x**2 + y**2) + U
#     else:
#         raise RuntimeError("The input combination is not viable.")


# def get_cr3bp_hamiltonian(mu=None, state=None, conj_mom=False):


#     if conj_mom and state is None:
#         x, y, _, px, py, pz, mu = sp.symbols("x y z px py pz par[0]")
#         Ubar = get_u_bar()
#         return sp.simplify(0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar)
#     elif conj_mom and state is not None:
#         Ubar = get_u_bar(mu, state[0:3])
#         x, y, _, px, py, pz = state
#         return 0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar
#     elif not conj_mom and state is None:
#         x, y, _, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
#         Ubar = get_u_bar()
#         return sp.simplify(0.5 * (vx**2 + vy**2 + vz**2) + Ubar)
#     elif not conj_mom and state is not None:
#         Ubar = get_u_bar(mu, state[0:3])
#         x, y, _, vx, vy, vz = state
#         return 0.5 * (vx**2 + vy**2 + vz**2) + Ubar
#     else:
#         raise RuntimeError("The input is malformed")
