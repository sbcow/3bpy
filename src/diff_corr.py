import numpy as np  # type: ignore

from src.hamiltonian_operations import (
    get_cr3bp_hamiltonian,
    get_hamiltonian_state_derivative,
    get_compiled_hamiltonian_state_derivative,
)


def corrector(ta, x0):
    """
    Performs a single natural parameter correction to the initial state with a fixed period. This method is very
    limited and unflexible. The method follows the simple relation:

    $\delta x_0 = - (\Phi_T - I_6)^{-1} \cdot \delta x_T$

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: np.ndarray
        Initial state corresponding to propagated Taylor-adaptive integrator

    Returns
    -------
    hy.taylor_adaptive
        A corrected Taylor-adaptive integrator object that has been propagated
    np.ndarray
        Initial state corresponding to the corrected, propagated Taylor-adaptive integrator
    float
        The norm of the error between the final and initial state

    """

    period = ta.time
    Phi_T = ta.state[6:].reshape((6, 6))
    dx_T = ta.state[:6] - x0
    dx0 = -np.linalg.inv(Phi_T - np.eye(6)) @ dx_T
    x0 += dx0
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(period)
    return ta, x0, np.linalg.norm(ta.state[:6] - x0)


def corrector_phasing(ta, x0, conj_mom=False):
    """
    Performs a single natural parameter correction to the initial state with an additional phasing equation and
    variable period. This method employs the following relation:

    $ A = \begin{bmatrix} A & f \\ f^T & 0 \\ \end{bmatrix}$
    $\delta x_0 = -A^{-1} \cdot \delta x_T$

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: list
        Initial state corresponding to propagated Taylor-adaptive integrator
    conj_mom: bool
        A boolean determining whether velocities or conjugate momenta are used

    Returns
    -------
    hy.taylor_adaptive
        A corrected Taylor-adaptive integrator object that has been propagated
    np.ndarray
        Initial state corresponding to the corrected, propagated Taylor-adaptive integrator
    float
        The norm of the error between the final and initial state

    """

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


def diff_corr(ta, x0, tol=1e-10, max_iter=100):
    """
    Performs a loop over numerous corrector() runs.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: np.ndarray
        Initial state corresponding to propagated Taylor-adaptive integrator
    tol: float
        The tolerance for the corrector to return
    max_iter: float
        The maximum number of iterations allowed

    Returns
    -------
    hy.taylor_adaptive
        A corrected Taylor-adaptive integrator object that has been propagated
    np.ndarray
        Initial state corresponding to the corrected, propagated Taylor-adaptive integrator
    float
        The norm of the error between the final and initial state

    See also
    --------
    corrector()

    """
    err = 1
    it = 0
    while err > tol:
        ta, x0, err = corrector(ta, x0)
        it += 1
        if it > max_iter:
            raise RuntimeError(f"Correction didn't converge. Error is still: {err}")
    return ta, x0, err


def diff_corr_phasing(ta, x0, tol=1e-10, max_iter=100, conj_mom=False):
    """
    Performs a loop over numerous corrector_phasing() runs.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: np.ndarray
        Initial state corresponding to propagated Taylor-adaptive integrator
    tol: float
        The tolerance for the corrector to return
    max_iter: float
        The maximum number of iterations allowed
    conj_mom: bool
        A boolean determining whether velocities or conjugate momenta are used

    Returns
    -------
    hy.taylor_adaptive
        A corrected Taylor-adaptive integrator object that has been propagated
    np.ndarray
        Initial state corresponding to the corrected, propagated Taylor-adaptive integrator
    float
        The norm of the error between the final and initial state

    See also
    --------
    corrector_phasing()

    """
    err = 1
    it = 0
    while err > tol:
        ta, x0, err = corrector_phasing(ta, x0, conj_mom=conj_mom)
        it += 1
        if it > max_iter:
            raise RuntimeError(f"Correction didn't converge. Error is still: {err}")
    return ta, np.insert(x0, 6, ta.time), err


def diff_corr_pa(
    ta, x0, t0, ds, dx, taux, taut, tol=1e-12, max_iter=100, conj_mom=False, verbose=False
):
    """
    Performs a Pseudo-arc correction scheme.

    Notes
    -----
    $ A = \begin{bmatrix} A & f \\ f^T & 0 \\ \tau_x & \tau_T \end{bmatrix}$ where $\tau_i$ refers
    to the tangential vector in the i direction (x being state and T being period)
    $[dx, dT] = (A^T \cdot A)^{-1} \cdot A^T \cdot err $ where err are the state, pointing, and
    phasing error as defined by the 8 equations.

    The tangent vector is passed from the predictor step.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: np.ndarray
        Initial state corresponding to propagated Taylor-adaptive integrator
    t0: float
        Initial time corresponding to initial state
    ds: float
        The initial variation used for the continuation
    dx: np.ndarray
        The difference between the original and predicted state
    taux: np.ndarray
        The state components of the tangent vector to the initial state
    taut: float
        The time component of the tangent vector to the initial state
    tol: float
        The tolerance for the corrector to return
    max_iter: float
        The maximum number of iterations allowed
    conj_mom: bool
        A boolean determining whether velocities or conjugate momenta are used
    verbose: bool
        Determines the verbosity of the output

    Returns
    -------
    hy.taylor_adaptive
        A corrected Taylor-adaptive integrator object that has been propagated
    np.ndarray
        Initial state and time (7x1) corresponding to the corrected, propagated Taylor-adaptive integrator
    float
        The norm of the state, pointing, and pseudo error between the final and initial state

    See also
    --------

    predictor_pa()

    """
    flag_tol = False

    curr_dx = dx[:6]
    curr_dT = dx[6]

    # Linearization point
    curr_x = x0 + curr_dx.T
    curr_T = t0 + curr_dT

    cf_f = get_compiled_hamiltonian_state_derivative(conj_mom=conj_mom)

    for i in range(max_iter):

        # 1 - We compute the state transition matrix Phi (integrating the full EOM from curr_x for t_final)
        ta.time = 0.0
        ta.state[:] = curr_x.tolist() + np.eye(6).reshape((36,)).tolist()
        _ = ta.propagate_until(curr_T)
        Phi = ta.state[6:].reshape((6, 6))

        # 2 - We compute the dynamics f (at the initial state)
        f_dyn0 = cf_f(x0, pars=[ta.pars[0]]).reshape(-1, 1)
        f_dynT = cf_f(ta.state[:6], pars=[ta.pars[0]]).reshape(-1, 1)

        # 3 - Assemble the function (the full nonlinear system) value at the current point
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
        # 4 - Assemble the matrix A (gradient)
        A = np.concatenate((Phi - np.eye(6), f_dynT.T))
        A = np.concatenate((A, np.insert(f_dynT, -1, 0).reshape((-1, 1))), axis=1)
        # add the tau row
        tmp = np.insert(taux, 6, taut).reshape(1, -1)
        A = np.concatenate((A, tmp))

        # 5 - Solve for new y = [dx, dT]
        y = (np.linalg.inv(A.T @ A) @ A.T @ b).reshape(
            -1,
        )
        curr_dx += y[:6]
        curr_dT += y[6]
        curr_x = x0 + curr_dx.T
        curr_T = t0 + curr_dT
    if flag_tol and verbose:
        print("Exit condition - Accuracy")
    elif verbose:
        print("Exit condition - Maximum Iterations")
    if verbose:
        print("Accuracy achieved: ", toterror)
        print("Iterations: ", i)

    return ta, np.insert(curr_x.T, 6, curr_T), toterror


def predictor_pa(ta, x0, t0, ds, conj_mom=False):
    """
    Performs a Pseudo-arc predictor step.

    Notes
    -----

    $\tau_T$ is assumed to be 1. $\tau_x$ ($\frac{dx}{ds}$) then becomes $(A_{6x6}^T \cdot
    A_{6x6})^{-1} \cdot A_{6x6}^T \cdot A_{7x1} \cdot \tau_T$
    $ A = \begin{bmatrix} A & f \\ f^T & 0 \\ \tau_x & \tau_T \end{bmatrix}$ where $\tau_i$ refers
    to the tangential vector in the i direction (x being state and T being period)
    $[dx, dT] = (A^T \cdot A)^{-1} \cdot A^T \cdot err $ where err are the state, pointing, and
    phasing error as defined by the 8 equations.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    x0: np.ndarray
        Initial state corresponding to propagated Taylor-adaptive integrator
    t0: float
        Initial time corresponding to initial state
    ds: float
        The initial variation used for the continuation
    conj_mom: bool
        A boolean determining whether velocities or conjugate momenta are used

    Returns
    -------
    np.ndarray
        Initial state and time (7x1) corresponding to the corrected, propagated Taylor-adaptive integrator
    np.ndarray
        The state elements of the tangent vector
    float
        The period element of the tangent vector

    See also
    --------

    diff_corr_pa()

    """
    # 1 - We compute the state transition matrix Phi (integrating the full EOM for t_final)
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(t0)
    Phi = ta.state[6:].reshape((6, 6))

    # 2 - We compute the dynamics f (at zero, but for periodic orbits this is the same at T)
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f_dyn = get_hamiltonian_state_derivative(
        H, ta.pars[0], x0, conj_mom=conj_mom
    ).reshape(-1, 1)

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

    # 7 - Predict an initial guess
    dx = np.linalg.inv(A.T @ A) @ (A.T @ b)
    return dx, taux, tauT


def predictor(ta, cont_param=0, variation=1e-4, conj_mom=True):
    """
    Performs a natural parameter predictor step.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    cont_param: int
        The column corresponding to the state element that is to be used for the prediction
    variation: float
        The initial variation used for the continuation
    conj_mom: bool
        A boolean determining whether velocities or conjugate momenta are used

    Returns
    -------
    np.ndarray
        Initial state (6x1) difference between the predicted and original state

    See also
    --------

    corrector()
    diff_corr()

    """
    Phi = ta.state[6:].reshape((6, 6))
    state_T = ta.state[:6]
    # Compute the dynamics from its expressions
    H = get_cr3bp_hamiltonian(conj_mom=True)
    f_dyn = get_hamiltonian_state_derivative(H, ta.pars[0], state_T, conj_mom=conj_mom).reshape(-1, 1)
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
    ta,
    new_ic=None,
    new_period=None,
    cont_param=None,
    ds=1e-4,
    verbose=True,
    corr_max_iter=100
):
    """
    Performs a Pseudo-arc continuation predictor-corrector step.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    new_ic: np.ndarray
        The periodic orbit initial state used
    new_period: float
        The period of the periodic orbit
    ds: float
        The variation used for the continuation
    verbose: bool
        Defines the verbosity of the output
    corr_max_iter: uint
        Defines the number of correction steps allowed

    Returns
    -------
    np.ndarray
        The periodic orbit initial state used
    float
        The period of the periodic orbit
    float
        The variation used for the continuation
    np.ndarray
       if err < 1e-12: A state history of the continued periodic orbit
    None
       if err > 1e-12 or Error: No output is created

    See also
    --------

    diff_corr_pa()
    predictor_pa()

    """

    out = None
    if new_ic is not None and new_period is not None and cont_param is None:
        predicted, taux, tauT = predictor_pa(ta, new_ic, new_period, ds, conj_mom=True)
    elif new_ic is None and new_period is None and cont_param is not None:
        predicted = predictor(ta, cont_param, ds, conj_mom=True)
    else:
        raise RuntimeError(f"Combination of inputs not valid.")
    # We call the corrector, if it fails (A^tA goes singular) we decrease the pseudo arc-length and try again
    try:

        if new_ic is not None and new_period is not None and cont_param is None:
            ta, curr_xT, err = diff_corr_pa(
                ta,
                new_ic,
                new_period,
                ds,
                predicted[:, 0],
                taux,
                tauT,
                max_iter=corr_max_iter,
                tol=1e-15,
                conj_mom=True,
            )
        elif new_ic is None and new_period is None and cont_param is not None:
            ta, curr_xT, err = diff_corr_phasing(
                ta,
                new_ic,
                tol=1e-15,
                max_iter=corr_max_iter,
                conj_mom=True,
            )
        else:
            raise RuntimeError(f"Combination of inputs not valid.")
        # Log
        if verbose:
            print("ds:", ds, "err:", err)
        # Accept the step
        if err < 1e-12:
            new_ic = curr_xT[:6]
            new_period = curr_xT[6]
            if verbose:
                print("Converged - increase ds")
            ds *= 1.1
            if ds > 0.05:
                ds = 0.05
            # Reset the state
            ta.time = 0
            ta.state[:] = new_ic.tolist() + np.eye(6).reshape((36,)).tolist()
            # Time grid
            t_grid = np.linspace(0, new_period, 2000)
            # Propagate
            out = ta.propagate_grid(t_grid)[5]
        # Reject the step
        else:
            ds /= 1.05
            if verbose:
                print("Low Precision - reducing ds")
    # The (A^T A) matrix was likely singular, we reduce the step ad try again.
    except:
        if verbose:
            print("Singular matrix - reducing ds")
        ds /= 1.05

    return new_ic, new_period, ds, out

def find_po_family(
    ta,
    new_ic,
    new_period,
    ds : float,
    max_iter = 100,
    corr_max_iter=100,
    verbose = True,
):
    """
    Performs Pseudo-arc continuation in a loop to find a family of periodic orbits.

    Parameters
    ----------
    ta: hy.taylor_adaptive
        A Taylor-adaptive integrator object that has been propagated
    new_ic: np.ndarray
        The periodic orbit initial state used
    new_period: float
        The period of the periodic orbit
    ds: float
        The variation used for the continuation
    max_iter: float
        The maximum number of times to continue the family
    corr_max_iter: float
        The maximum number of times to correct an orbit
    verbose: bool
        Defines the verbosity of the output

    Returns
    -------
    dict[np.ndarray, None]
        A dict of state history arrays and None objects
    dict[np.ndarray]
        A dict of periods corresponding to each iteration

    See also
    --------

    perform_predictor_corrector_step()

    """
    out = {}
    periods = {}
    for i in range(max_iter):
        new_ic, new_period, ds, out[i] = perform_predictor_corrector_step(ta, new_ic=new_ic,
                                                                          new_period=new_period,
                                                                          ds=ds, verbose=verbose,
                                                                          corr_max_iter=corr_max_iter)
        periods[i] = new_period if out is not None else None
    return out, periods
