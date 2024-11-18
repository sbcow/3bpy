from initial_exploration_ross import *
import numpy as np
import copy
import sympy as sp

##########################
### Initial conditions ###
##########################

# New mu parameter (no reason to change, just came out playing)
mu = 0.01215057
# Initial guess for the integration time (will eventually converge to a period)
t_final = 3.

buffer = 0.002
xL1, xL2, xL3, (xL4, yL4), (xL5, yL5) = find_lagrange_points(mu)

# init_cond = [xL1 * 1.001, 0.,0.,0.,  -9.26435468e-02, 0.]
# In vel form
# init_cond = [ xL1 - 0.0001057585798973637, 0., 0., 0., -8.85435468e-04, 0.]
# init_cond = [ xL1 - 0.0001057585798973637, 0., 0., 0., -0.8376948795, 0.]
init_cond_vel = [ xL1 + 0.0001057585798973637, 0., 0., 0., -8.85435468e-04, 0.]
init_cond_cm = convert_vel_to_conj_mom(init_cond_vel.copy())
init_cond = init_cond_cm

init_cond_var = np.eye(6).reshape((36,)).tolist()

####################
### Define event ###
####################

xaxis_crossings = []
def callback_root(ta, d_sgn):#, time):#, d_sgn):
    print(f"y-coord is zero at t, x={ta.time}, {ta.d_output[0]}")
    xaxis_crossings.append((ta.time, ta.d_output[0]))

    # Stop integration
    return False

y = hy.make_vars("y")

t_ev = hy.t_event(
    # The event equation.
    y,
    # The callback.
    callback = callback_root,
    direction = hy.event_direction.positive)

########################
### Setup simulation ###
########################

H = get_cr3bp_hamiltonian(conj_mom=True)
ta = get_ta_var(init_cond, init_cond_var, conj_mom=True, event=t_ev)

E = get_cr3bp_hamiltonian(mu=mu, state=init_cond_vel, conj_mom=False)
nsteps = 2000
epochs = np.linspace(0, t_final, nsteps)

x0 = np.array(init_cond)
ta.time = 0.0
ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
ta.pars[0] = mu
out = ta.propagate_grid(epochs)
out = out[5]
half_period = xaxis_crossings[0][0]

new_times = np.linspace(0.0, 2 * half_period, 2000)

# ta_original = copy.deepcopy(ta)
# x0_original = copy.deepcopy(x0)

# No event
ta_noevent = get_ta_var(x0.tolist(), init_cond_var, conj_mom=True)
ta_noevent.time = 0.0
ta_noevent.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
ta_noevent.pars[0] = mu
out_init = ta_noevent.propagate_grid(new_times)
out_init = out_init[5]

# With x0 += dx0 / 1e4, it converges quickly and stops at ~1e-6
#Only x0[4] += dx0[4] /1e2 also converges quickly to ~1e-6

def corrector(ta, x0):

    state_T = ta.state[:6]
    Phi_T = ta.state[6:].reshape((6, 6))
    dx_T = state_T - x0
    print("error was:", np.linalg.norm(dx_T))
    dx0 = np.linalg.inv(np.eye(6) - Phi_T)@dx_T
    # print("condition number is:", np.linalg.cond(Phi_T))
    # x0[4] += dx0[4] / 1e2# * np.linalg.norm(dx_T) * 10
    x0[4] += dx0[4]# / 1e2
    ta.pars[0] = mu
    ta.time = 0.0
    ta.state[:] = x0.tolist() + np.eye(6).reshape((36,)).tolist()
    _ = ta.propagate_until(2 * half_period)
    state_T = ta.state[:6]
    dx_T = state_T - x0
    print("new error is:", np.linalg.norm(dx_T))
    return ta, x0

ta = ta_noevent
for _ in range(5):
    ta, x0 = corrector(ta, x0)

ta_periodic = ta
x0_periodic = x0
ta_periodic.time = 0.0
ta_periodic.state[:] = x0_periodic.tolist() + np.eye(6).reshape((36,)).tolist()
ta_periodic.pars[0] = mu
out_periodic = ta_periodic.propagate_grid(new_times)
out_periodic = out_periodic[5]

# Define relevant positions in rotating frame
m1_pos = np.array([-mu, 0, 0])
m2_pos = np.array([1 - mu, 0, 0])
L1_pos = np.array([xL1, 0, 0])
L2_pos = np.array([xL2, 0, 0])
L3_pos = np.array([xL3, 0, 0])
L4_pos = np.array([xL4, yL4, 0])
L5_pos = np.array([xL5, yL5, 0])

# Find the forbidden region
xx = np.linspace(xL1 - buffer, xL1 + buffer, 2000)
yy = np.linspace(-buffer, buffer, 2000)
# xx = np.linspace(-1.5, 1.5, 2000)
# yy = np.linspace(-1.5, 1.5, 2000)
x_grid, y_grid = np.meshgrid(xx, yy)
symbs = ["x", "y", "z", "xdot", "ydot", "zdot"]
potentials = get_u_bar(mu, (x_grid, y_grid, np.zeros(np.shape(x_grid))))

################
### Plotting ###
################

plt.figure()
# s/c trajectory
plt.plot(out_init[:, 0], out_init[:, 1], linestyle="-", linewidth=1, c='r')
plt.scatter(out_init[-1, 0], out_init[-1, 1], s=5)
# plt.plot(out[:, 0], out[:, 1], linestyle="-", linewidth=1, c='b')
# plt.scatter(out[-1, 0], out[-1, 1], s=5)
plt.plot(out_periodic[:, 0], out_periodic[:, 1], linestyle="-", linewidth=1, c='k')
plt.scatter(out_periodic[-1, 0], out_periodic[-1, 1], s=5)
plt.scatter(xaxis_crossings[0][1], 0, s=20)

# masses
plt.scatter(-mu, 0, c="r", s=20)  # m1
plt.scatter(1 - mu, 0, c="r", s=20)  # m2

# Lagrange points
plt.scatter(xL1, 0, c="k", s=10)  # m2
plt.scatter(xL2, 0, c="k", s=10)  # m2
plt.scatter(xL3, 0, c="k", s=10)  # m2
plt.scatter(xL4, yL4, c="k", s=10)  # m2
plt.scatter(xL5, yL5, c="k", s=10)  # m2

# zero velocity curve
plt.imshow((potentials >= E).astype(int),
    extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
    origin="lower",
    cmap="Greens"
    )

plt.title(f"Top-down view - mu: {mu} - Rotating frame")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.xlim([xL1 - buffer, xL1 + buffer])
plt.ylim([0.0 - buffer, 0.0 + buffer])
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
plt.tight_layout()

plt.show()