# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# These masses represent the Earth-Moon system
m_1 = 5.974e24  # kg
m_2 = 7.348e22  # kg
mu = m_2 / (m_1 + m_2)

# Initial conditions
x_0 = 1 - mu
y_0 = 0.0455
z_0 = 0
vx_0 = -0.5
vy_0 = 0.5
vz_0 = 0

# Then stack everything together into the state vector
r_0 = np.array((x_0, y_0, z_0))
v_0 = np.array((vx_0, vy_0, vz_0))
Y_0 = np.hstack((r_0, v_0))

def nondim_cr3bp(t, Y):
    """Solve the CR3BP in nondimensional coordinates.

    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.

    The solution is parameterized on $\\mu$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    # Define the derivative vector
    Ydot = np.zeros_like(Y)
    Ydot[:3] = Y[3:]

    sigma = np.sqrt(np.sum(np.square([x + mu, y, z])))
    psi = np.sqrt(np.sum(np.square([x - 1 + mu, y, z])))
    Ydot[3] = (
        2 * ydot
        + x
        - (1 - mu) * (x + mu) / sigma**3
        - mu * (x - 1 + mu) / psi**3
    )
    Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    Ydot[5] = -(1 - mu) / sigma**3 * z - mu / psi**3 * z
    return Ydot


# Set time and integrate
t_0 = 0  # nondimensional time
t_f = 20  # nondimensional time
t_points = np.linspace(t_0, t_f, 1000)
sol = solve_ivp(nondim_cr3bp, [t_0, t_f], Y_0, t_eval=t_points)

Y = sol.y.T
r = Y[:, :3]  # nondimensional distance
v = Y[:, 3:]  # nondimensional velocity

# Plot
x_2 = (1 - mu) * np.cos(np.linspace(0, np.pi, 100))
y_2 = (1 - mu) * np.sin(np.linspace(0, np.pi, 100))
fig, ax = plt.subplots(figsize=(5, 5), dpi=96)

# Plot the orbits
ax.plot(r[:, 0], r[:, 1], "r", label="Trajectory")
ax.axhline(0, color="k")
ax.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
ax.plot(-mu, 0, "bo", label="$m_1$")
ax.plot(1 - mu, 0, "go", label="$m_2$")
ax.plot(x_0, y_0, "ro")
ax.set_aspect("equal")
plt.show()



