import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from src.richardson.halo_sv import halo_sv

y = np.zeros((6, 1))
ampl_z = 0.1#  * AU
ilp = 2
hclass = 2


# -----------------
# earth-moon system
# -----------------

# # gravitational constant of earth (kilometers^3/seconds^2)
# mu1 = 398600.4415
#
# # gravitational constant of the moon (kilometers^3/seconds^2)
# mu2 = 4902.8
#
# # normalized gravitational constant
# mu = mu2 / (mu1 + mu2)
mu = 0.01215057

# distance between primary and secondary bodies (kilometers)
dist = 1

# -------------------------------------------------
# compute initial guess for normalized state vector
# and orbital period using richardson's algorithm
# -------------------------------------------------

r_halo, v_halo, period = halo_sv(amp_z = ampl_z, t=0.0, hclass=hclass, dist=dist, mu=mu, ilp=ilp)
state = np.concatenate((r_halo, v_halo))
state_form = [elem for elem in state]

print(state_form)
print(period)
