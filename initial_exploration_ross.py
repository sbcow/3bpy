import numpy as np # type: ignore
import sympy as sp
import heyoka as hy
import matplotlib.pyplot as plt


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
def get_u_bar(mu=None, position=None):

    if position is None and mu is None:
        x, y, z, mu = sp.symbols("x y z par[0]")
        mu2 = mu
        mu1 = 1-mu
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
        x, y, _, px, py, pz, mu = sp.symbols("x y z px py pz par[0]")
        Ubar = get_u_bar()
        return sp.simplify(0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar)
    elif conj_mom and state is not None:
        Ubar = get_u_bar(mu, state[0:3])
        x, y, _, px, py, pz = state
        return 0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar
    elif not conj_mom and state is None:
        x, y, _, vx, vy, vz, mu = sp.symbols("x y z vx vy vz par[0]")
        Ubar = get_u_bar()
        return sp.simplify(0.5 * (vx**2 + vy**2 + vz**2) + Ubar)
    elif not conj_mom and state is not None:
        Ubar = get_u_bar(mu, state[0:3])
        x, y, _, vx, vy, vz = state
        return 0.5 * (vx**2 + vy**2 + vz**2) + Ubar
    else:
        raise RuntimeError("The input is malformed")

# def get_u_bar(mu, position=None):

#     mu2 = mu
#     mu1 = 1-mu
#     if position is None:
#         x, y, z = sp.symbols("x y z")
#         r1 = sp.sqrt((x + mu2) ** 2 + y**2 + z**2)
#         r2 = sp.sqrt((x - mu1) ** 2 + y**2 + z**2)

#         U = (-mu1 / r1) - (mu2 / r2) - (0.5 * mu1 * mu2)
#         return -0.5 * (x**2 + y**2) + U

#     elif position is not None:
#         x, y, z = position
#         r1 = np.sqrt((x + mu2) ** 2 + y**2 + z**2)
#         r2 = np.sqrt((x - mu1) ** 2 + y**2 + z**2)

#         U = -mu1 / r1 - mu2 / r2 - 0.5 * mu1 * mu2
#         return -0.5 * (x**2 + y**2) + U
#     else:
#         raise RuntimeError("The input combination is not viable.")



# def get_cr3bp_hamiltonian(mu, state=None, conj_mom=False):


#     if conj_mom and state is None:
#         Ubar = get_u_bar(mu)
#         x, y, _, px, py, pz = sp.symbols("x y z px py pz")
#         return sp.simplify(0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar)
#     elif conj_mom and state is not None:
#         Ubar = get_u_bar(mu, state[0:3])
#         x, y, _, px, py, pz = state
#         return 0.5 * ((px + y) ** 2 + (py - x) ** 2 + pz**2) + Ubar
#     elif not conj_mom and state is None:
#         Ubar = get_u_bar(mu)
#         x, y, _, vx, vy, vz = sp.symbols("x y z vx vy vz")
#         return sp.simplify(0.5 * (vx**2 + vy**2 + vz**2) + Ubar)
#     elif not conj_mom and state is not None:
#         Ubar = get_u_bar(mu, state[0:3])
#         x, y, _, vx, vy, vz = state
#         return 0.5 * (vx**2 + vy**2 + vz**2) + Ubar
#     else:
#         raise RuntimeError("The input is malformed")

def get_jacobi_integral(mu, state=None, conj_mom=False):
    E = get_cr3bp_hamiltonian(mu, state, conj_mom)
    return -2 * E


def get_hamiltonian_state_derivative(H, conj_mom=False):

    if conj_mom:
        x, y, z, px, py, pz = sp.symbols("x y z px py pz")
        xdot = sp.diff(H, px)
        ydot = sp.diff(H, py)
        zdot = sp.diff(H, pz)
        pxdot = -sp.diff(H, x)
        pydot = -sp.diff(H, y)
        pzdot = -sp.diff(H, z)
        return np.array([xdot, ydot, zdot, pxdot, pydot, pzdot])
    else:
        x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz")
        xdot = sp.diff(H, vx)
        ydot = sp.diff(H, vy)
        zdot = sp.diff(H, vz)
        vxdot = -sp.diff(H, x)
        vydot = -sp.diff(H, y)
        vzdot = -sp.diff(H, z)
        return np.array([xdot, ydot, zdot, vxdot, vydot, vzdot])


def convert_sympy_to_hy(state_derivative):
    return [hy.from_sympy(i) for i in state_derivative]


def get_ta(H, init_cond, batch_mode=False, conj_mom=False):
    if not conj_mom:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    else:
        # Create the symbolic variables in heyoka.
        x, y, z, xdot, ydot, zdot  = hy.make_vars("x", "y", "z", "px", "py", "pz")

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


def get_mu_bar(mu, xL):
    return mu / np.abs(xL - 1 + mu) ** 3 + (1 - mu) / np.abs(xL + mu) ** 3


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


def find_lagrange_points(mu):

    # Find L point
    pL1 = np.polynomial.Polynomial([1, -1 * (3 - mu), (3 - 2 * mu), -mu, 2 * mu, -mu])
    rootsL1 = np.roots(pL1.coef)
    pL2 = np.polynomial.Polynomial([1, -1 * (3 - mu), (3 - 2 * mu), -mu, -2 * mu, -mu])
    rootsL2 = np.roots(pL2.coef)
    for it in range(len(rootsL1)):
        if np.isreal(rootsL1[it]):
            gammaL1 = rootsL1[it].real
        if np.isreal(rootsL2[it]):
            gammaL2 = rootsL2[it].real
    xL1 = (1 - mu) - gammaL1
    xL2 = (1 - mu) + gammaL2
    xL3 = -mu - (1 - 7 * mu / 12)  # Approximation wikipedia

    # By convention, L4 and L5 lie at:
    xL4, yL4 = (0.5 - mu, np.sqrt(3) / 2)
    xL5, yL5 = (0.5 - mu, -np.sqrt(3) / 2)

    return xL1, xL2, xL3, (xL4, yL4), (xL5, yL5)


###############################
### Differential correction ###
###############################

def liadifcor(x0g, show=0):
# [x0,t1]=liadifcor(x0g,show)
# 
# This is the differential correction routine to create planar periodic 
# Liapunov orbits about L1,L2, or L3 in the CR3BP. It keeps the initial 
# x value constant and varies the y-velocity value.
# 
# output: x0  = initial state on the Liapunov (on the xz-plane) in nondim.
# 	CR3BP coords.
#   t1  = half-period of Liapunov orbit in nondim. CR3BP time 
# 
# input: x0g  = first guess of initial state on the Liapunov orbit
#  show = 1 to plot successive orbit  (default=0)
# 
# ------------------------------------------------------------------------
# CR3BP with the LARGER MASS, m1 to the left of the origin at (-mu,0)
# and m2, or the planet (ie. Earth), is at (1 - mu, 0)
# 
#                L4
# -L3----m1--+-------L1--m2--L2-
#                L5
# 
# Shane Ross (revised 8.28.97)
    global mu

    t0 = 0
    dxdot1 = 1
    MAXdxdot1 = 1.e-12  # measure of precision of perpendicular x-axis crossing
    attempt = 0
    MAXattempt = 25

    while abs(dxdot1) > MAXdxdot1:
        if attempt > MAXattempt:
            ERROR = 'OCCURED'
            break

        t1, xx1 = find0(x0g)  # find x-axis crossing
        attempt += 1

        x1, y1 = xx1[0], xx1[1]
        dxdot1, ydot1 = xx1[2], xx1[3]

        x, t, phi_t1, PHI = PHIget(x0g, t1)

        if show == 1:
            plotX(x)
            plt.plot(a * x[0, 0], a * x[0, 1], 'wo')
            plt.axis([min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1])])
            plt.axis('equal')

        mu2 = 1 - mu
        r1 = ((x1 + mu) ** 2 + y1 ** 2)
        r2 = ((x1 - mu2) ** 2 + y1 ** 2)
        rho1 = 1 / r1 ** 1.5
        rho2 = 1 / r2 ** 1.5

        omgx1 = -(mu2 * (x1 + mu) * rho1) - (mu * (x1 - mu2) * rho2) + x1

        xdotdot1 = (2 * ydot1 + omgx1)

        C1 = phi_t1[2, 3]
        C2 = phi_t1[1, 3]
        C3 = (C1 - (1 / ydot1) * xdotdot1 * C2)
        C4 = np.linalg.inv(C3) * (-dxdot1)
        dydot0 = C4

        x0g[3] = x0g[3] + dydot0

    x0 = x0g
    return x0, t1



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