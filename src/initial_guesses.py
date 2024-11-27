import numpy as np  # type: ignore

from src.hamiltonian_operations import convert_vel_to_conj_mom, get_cr3bp_jacobian


def get_po_ig(xL, mu_val, use_real=True, disp=1e-4, direction=True, conj_mom=False):
    """
    Attain an initial guess for a periodic orbit using the State Transition Matrix at a Lagrange
    point.


    Parameters
    ----------
    xL: float
        The x-coordinate of the Lagrange point
    mu_val: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    use_real: bool
        Whether to use the real or imaginary eigenvalue
    disp: float
        The displacement along the direction of the chosen eigen vector
    direction: bool
        The direction of the displacement
    conj_mom: bool
        Determines whether or not the conjugate momenta formulation is used or not.


    Returns
    -------
    list
        A list of the initial guess state elements
    float
        The initial guess of the period
    """
    if not conj_mom:
        state = np.array([xL, 0, 0, 0, 0, 0])
    else:
        state = convert_vel_to_conj_mom(np.array([xL, 0, 0, 0, 0, 0]))
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
    po_ic = (
        (state + first_oscillatory_direction * disp).tolist()
        if direction is True
        else (state - first_oscillatory_direction * disp).tolist()
    )

    return po_ic, po_period_guess


def get_richardson_halo_ig():
    pass
