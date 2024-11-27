import numpy as np
import time
from matplotlib.pylab import plt
import pickle 
import os

from src.hamiltonian_operations import find_lagrange_points, convert_vel_to_conj_mom
from src.diff_corr import diff_corr_phasing, find_po_family
from src.initial_guesses import get_po_ig
from src.ta_construction import get_ta_var
from src.miscellaneous import plot_zero_vel_curves


####################
### Define input ###
####################

# Get periodic orbit initial guess
mu = 0.01215057
# We find the postiion of the lagrangian points
xl_list = find_lagrange_points(mu)
xL1, xL2, xL3, (xL4, yL4), (xL5, yL5) = (
    xl_list[1],
    xl_list[2],
    xl_list[0],
    (xl_list[3][0], xl_list[3][1]),
    (xl_list[4][0], xl_list[4][1]),
)

xL = xL1

#####################
### Get variables ###
#####################

po_ic, po_period = get_po_ig(
    xL, mu, use_real=False, direction=False, conj_mom=True
)
init_cond_var = np.eye(6).reshape((36,)).tolist()
ta = get_ta_var(po_ic, ic_var=init_cond_var, conj_mom=True)

nsteps = 2000
epochs = np.linspace(0, po_period, nsteps)

ta.pars[0] = mu
ta.time = 0
ta.state[:] = po_ic + np.eye(6).reshape((36,)).tolist()
out2 = ta.propagate_grid(epochs)
out2 = out2[5]

ta, ic, err = diff_corr_phasing(ta, po_ic, tol=1e-12, max_iter=100, conj_mom=True)
t_final = ic[6]
ic = ic[:6]

timesteps = np.linspace(0, t_final, 2000)
ta.time = 0.0
ta.state[:] = ic.tolist() + init_cond_var
out3 = ta.propagate_grid(timesteps)
out3 = out3[5]

# Starting point
new_ic = ic
new_period = t_final

#######################################
### Perform Pseudo-Arc Continuation ###
#######################################

# Set to true to get more info on the iterations
out, periods = find_po_family(ta, np.array(po_ic), po_period, ds=1e-4, max_iter=200, corr_max_iter=100, verbose=True)

#################
### Plotting  ###
#################

plot = False
if plot:
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)

# Plot the lagrangian points and primaries
    buffer = 1.5
    ax.set_xlim(xL1 - buffer, xL1 + buffer)
    ax.set_ylim(-buffer, +buffer)
    ax.scatter(-mu, 0, c="k", s=100)
    ax.scatter((1 - mu), 0, c="k", s=30)
    ax.scatter(xL1, 0, c="r", s=10)
    ax.scatter(xL2, 0, c="r", s=10)
    ax.scatter(xL3, 0, c="r", s=10)
    ax.scatter(xL4, yL4, c="r", s=10)
    ax.scatter(xL5, yL5, c="r", s=10)
# # zero velocity curve
    plot_zero_vel_curves(ax, mu, new_ic, xL, buffer=buffer)
    ax.plot(out2[:, 0], out2[:, 1], linewidth=2, c="b")
    ax.scatter(out2[-1, 0], out2[-1, 1], s=5)
    ax.plot(out3[:, 0], out3[:, 1], linewidth=2, c="r")
    ax.scatter(out3[-1, 0], out3[-1, 1], s=5)

    for i in out.keys():
        ax.plot(out[i][:, 0], out[i][:, 1], "g", alpha=0.3) if out[i] is not None else None

    plt.show()

###################
### Saving data ###
###################

save_data = True
if save_data:
    test_id = 'test_3'
    direc = os.getcwd() + f'/data/{test_id}/'
    if not os.path.exists(direc):
        os.makedirs(direc)
    with open(direc + 'output_lyapunov.pkl', 'wb') as f:
        pickle.dump(out, f)
    with open(direc + 'periods.pkl', 'wb') as f:
        pickle.dump(periods, f)

#         
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
