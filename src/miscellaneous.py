import numpy as np  # type: ignore

from src.hamiltonian_operations import get_cr3bp_hamiltonian, get_u_bar


def plot_zero_vel_curves(ax, mu_val, ic, xL, buffer=1.5, resolution=2000):
    """
    Adds the zero velocity curves to a plot (a plt.image.AxesImage is added)

    Parameters
    ----------
    ax: plt.axes._axes.Axes
        The axis object for the curves to be added to
    mu_val: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    ic: np.ndarray
        The initial state of the orbit
    xL: float
        The x-coordinate of the Lagrange point
    buffer: float
        The buffer around xL for which to plot the velocity curve
    resolution: float
        The number of points for which to calculate the zero velocity condition

    Returns
    -------
    None
    """
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


def get_mu_bar(mu, xL):
    """
    Get $\bar{\mu}$ as defined by Dynamical Systems ..., Koon (2022).


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    xL: float
        The x-coordinate of the Lagrange point

    Returns
    -------
    float
        The $\bar{\mu}$ value
    """
    return mu / np.abs(xL - 1 + mu) ** 3 + (1 - mu) / np.abs(xL + mu) ** 3


def get_eigen_values_at_L(mu, xL):
    """
    Get the eigen values at a given Lagrange point.

    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    xL: float
        The x-coordinate of the Lagrange point

    Returns
    -------
    float
        $\lambda$ being the Real eigen value
    float
        $\nu$ being the Imaginary eigen value magnitude
    """
    mubar = get_mu_bar(mu, xL)
    lam = np.sqrt(0.5 * (mubar - 2 + np.sqrt(9 * mubar**2 - 8 * mubar)))
    nu = np.sqrt(-0.5 * (mubar - 2 - np.sqrt(9 * mubar**2 - 8 * mubar)))
    return lam, nu


def get_sigma_tau_constants(mu, xL):
    """
    Get the $\sigma$ and $\tau$ constants as defined by Dynamical Systems ..., Koon (2022).


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    xL: float
        The x-coordinate of the Lagrange point

    Returns
    -------
    float
        $\sigma$ being a derivative value related to $\lambda$
    float
        $\tau$ being a derivative value related to $\nu$
    """

    mubar = get_mu_bar(mu, xL)
    lam, nu = get_eigen_values_at_L(mu, xL)
    a = 1 + 2 * mubar
    b = mubar - 1

    sigma = 2 * lam / (lam**2 + b)
    tau = -(nu**2 + a) / (2 * nu)

    return sigma, tau


def get_eigenbasis_vectors(mu, xL):
    """
    Get the eigenvectors of the system as defined by Dynamical Systems ..., Koon (2022).


    Parameters
    ----------
    mu: float
        The mass ratio ($mu = m1 / (m1 + m2)$) of the system
    xL: float
        The x-coordinate of the Lagrange point

    Returns
    -------
    np.ndarray
        u1 corresponding to the first eigen vector
    np.ndarray
        u2 corresponding to the second eigen vector
    np.ndarray
        u corresponding to the third eigen vector
    np.ndarray
        v corresponding to the fourth eigen vector
    """
    lam, nu = get_eigen_values_at_L(mu, xL)
    sigma, tau = get_sigma_tau_constants(mu, xL)

    u1 = np.array([1, -sigma, lam, -lam * sigma])
    u2 = np.array([1, sigma, -lam, -lam * sigma])
    u = np.array([1, 0, 0, nu * tau])
    v = np.array([0, tau, nu, 0])
    return u1, u2, u, v
