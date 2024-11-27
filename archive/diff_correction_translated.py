import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Global variables
FORWARD = 1
tol = 1e-8
mu = 0.0121  # Earth-Moon

poly1 = [1, -1 * (3 - mu), (3 - 2 * mu), -mu, 2 * mu, -mu]

rt1 = np.roots(poly1)
gamma = None
for k in range(5):
    if np.isreal(rt1[k]):
        gamma = rt1[k]

xL = (1 - mu) - gamma

T = 2 * np.pi  # period of primaries in non-dimensional units

# L1 eigenbasis
mubar = mu / abs(xL - 1 + mu) ** 3 + (1 - mu) / abs(xL + mu) ** 3

a = 1 + 2 * mubar
b = mubar - 1

lam = np.sqrt(0.5 * (mubar - 2 + np.sqrt(9 * mubar ** 2 - 8 * mubar)))
nu = np.sqrt(-0.5 * (mubar - 2 - np.sqrt(9 * mubar ** 2 - 8 * mubar)))

sigma = 2 * lam / (lam ** 2 + b)
tau = -(nu ** 2 + a) / (2 * nu)

u1 = np.array([1, -sigma, lam, -lam * sigma])
u2 = np.array([1, sigma, -lam, -lam * sigma])
u = np.array([1, 0, 0, nu * tau])
v = np.array([0, tau, -nu, 0])

# Compute small periodic orbit (planar Lyapunov orbit)
tfinal = 2 * np.pi / nu
tspan = np.arange(0, tfinal, 0.001 * T)

XL = np.array([xL, 0, 0, 0])
displacement = 1e-3  # nondimensional displacement from L
X0 = XL + displacement * u  # initial condition
X0[3] = 0.9985 * X0[3]

def prtbp(t, x):
    global mu
    mu1 = 1 - mu  # mass of larger primary (nearest origin on left)
    mu2 = mu  # mass of smaller primary (furthest from origin on right)

    r3 = ((x[0] + mu2) ** 2 + x[1] ** 2) ** 1.5  # r: distance to m1, LARGER MASS
    R3 = ((x[0] - mu1) ** 2 + x[1] ** 2) ** 1.5  # R: distance to m2, smaller mass

    xdot = np.zeros(4)
    xdot[0] = x[2]
    xdot[1] = x[3]
    xdot[2] = x[0] - (mu1 * (x[0] + mu2) / r3) - (mu2 * (x[0] - mu1) / R3) + 2 * x[3]
    xdot[3] = x[1] - (mu1 * x[1] / r3) - (mu2 * x[1] / R3) - 2 * x[2]
    return xdot

sol = solve_ivp(prtbp, [0, tfinal], X0, method='RK45', t_eval=tspan, rtol=1e-6, atol=1e-8)

x = sol.y[0]
y = sol.y[1]

plt.plot(x, y, 'k', X0[0], X0[1], 'ro')
plt.title('Rotating Frame')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)

# Additional computations and plotting would follow here...

def energy(X, mu):
    mu1 = 1 - mu
    mu2 = mu

    Vsqrd = X[:, 2]**2 + X[:, 3]**2
    Ubar =  \
           - mu2 / np.sqrt((X[:, 0] - mu1)**2 + X[:, 1]**2) \
           - 0.5 * (X[:, 0]**2 + X[:, 1]**2) - 0.5 * mu1 * mu2

    E = 0.5 * Vsqrd + Ubar

    return E
