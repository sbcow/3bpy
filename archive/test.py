import sympy as sp
import time
from initial_exploration_ross import *

##########################
### Initial conditions ###
##########################

mu_val = 0.01215057
xl_list = find_lagrange_points(mu_val)
xL1, xL2, xL3, (xL4, yL4), (xL5, yL5) = (
    xl_list[1],
    xl_list[2],
    xl_list[0],
    (xl_list[3][0], xl_list[3][1]),
    (xl_list[4][0], xl_list[4][1]),
)
xL = xL1
buffer = 0.005
init_cond_vel = [xL, 0.0, 0.0, 0.0, 0.0, 0.0]
init_cond_cm = convert_vel_to_conj_mom(init_cond_vel.copy())
init_cond_var = np.eye(6).reshape((36,)).tolist()

# Get periodic orbit initial guess
po_ic, po_period = get_po_ig(xL, mu_val, use_real=True, direction=True)


################################
### Setup ta_var propagation ###
################################

ta_var = get_ta_var(po_ic, ic_var=init_cond_var, conj_mom=False)

nsteps = 2000
epochs = np.linspace(0, po_period, nsteps)

ta_var.pars[0] = mu_val
ta_var.time = 0
ta_var.state[:] = po_ic + np.eye(6).reshape((36,)).tolist()
out2 = ta_var.propagate_grid(epochs)
out2 = out2[5]

#######################################
### Perform differential correction ###
#######################################

ta_var, x0, err = diff_corr_phasing(ta_var, po_ic, tol=1e-12, max_iter=100)

########################################################
### Propagation of converged periodic Lyapunov orbit ###
########################################################
po_period_updated = ta_var.time
timesteps = np.linspace(0, po_period_updated, 2000)
ta_var.time = 0.0
ta_var.state[:] = x0.tolist() + init_cond_var
out3 = ta_var.propagate_grid(timesteps)
out3 = out3[5]

#######################################################
### Perform predictor-correction step and propagate ###
#######################################################

out = {}
no_of_cont = 100
cont_param = 0
variation = 1e-4
max_iter = 100
cont_type = 'phasing'
for i in range(no_of_cont):
    # ta_var, x0 = perform_predictor_corrector_step(ta_var, cont_param=0, with_phasing=True, tol=1e-12, variation=1e-6, direction=True)
    # ta_var, x0 = perform_predictor_corrector_step( ta_var, (ta_var, cont_param, variation), (), predictor_func=predictor, corrector_func=corrector_phasing, tol=1e-12, direction=True,)
    try:
        ta_var, x0, err = perform_predictor_corrector_step(
            ta_var,
            cont_type,
            (cont_param, variation),
            # predictor_func_args=(x0, po_period_updated, variation),
            # corrector_func_args=(x0, po_period_updated, variation),
            # corrector_func_args=(variation),
            tol=1e-12,
            direction=True,
            max_iter=max_iter
        )
        print("variation:", variation, "err:", err)
        variation*=1.2
        if variation >0.05:
            variation=0.05
        
        # Propagate
        timesteps = np.linspace(0, ta_var.time, 2000)
        ta_var.time = 0.0
        ta_var.state[:] = x0.tolist() + init_cond_var
        out[i] = ta_var.propagate_grid(timesteps)[5]
    except:
        print(f"Something went wrong, decreasing variation from {variation} to {variation / 1.1}")
        variation /= 1.1

################
### Plotting ###
################

fig, ax = plt.subplots()
# s/c trajectory
# plt.plot(out[:, 0], out[:, 1], linestyle="--", linewidth=1, c='k')
# plt.scatter(out[-1, 0], out[-1, 1], s=5)
plt.plot(out2[:, 0], out2[:, 1], linestyle="--", linewidth=1, c="b")
plt.scatter(out2[-1, 0], out2[-1, 1], s=5)
ax.plot(out3[:, 0], out3[:, 1], linestyle="--", linewidth=1, c="g")
ax.scatter(out3[-1, 0], out3[-1, 1], s=5)
for i in range(no_of_cont):
    ax.plot(out[i][:, 0], out[i][:, 1], linestyle="-", linewidth=1, c="g")
    # ax.scatter(out[i][-1, 0], out[i][-1, 1], s=5)

# # masses
ax.scatter(-mu_val, 0, c="r", s=20)  # m1
ax.scatter(1 - mu_val, 0, c="r", s=20)  # m2

# # Lagrange points
ax.scatter(xL, 0, c="k", s=10)  # m2
# # zero velocity curve
plot_zero_vel_curves(ax, mu_val, x0, xL, buffer=buffer)

ax.set_title(f"Top-down view - mu: {mu_val} - Rotating frame")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_xlim([xL - buffer, xL + buffer])
ax.set_ylim([0.0 - buffer, 0.0 + buffer])
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
plt.tight_layout()

plt.show()
