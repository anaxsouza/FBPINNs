"""
Created on Tue May 02 14:28:00 2022

@author: anaxsouza
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../fbpinns/')
sys.dont_write_bytecode = True
import problems
import losses
import boundary_conditions
import constants
import active_schedulers
import main
import models

#hyper-parameters

width = .7

P = problems.Cos_Cos2D_1(w=15, A=0)
subdomain_xs = constants.get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
                                 np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], 
                                [2*np.pi, 2*np.pi])

subdomain_xs_2 = constants.get_subdomain_xs([np.array([2,2,2,2]),
                                 np.array([2,2,2,2])], 
                                [2*np.pi, 2*np.pi])

subdomain_ws = constants.get_subdomain_ws(subdomain_xs, width)

subdomain_ws_2 = constants.get_subdomain_ws(subdomain_xs_2, width)

boundary_n = (1/P.w,)
y_n = (0,1/P.w)
batch_size = (900,900)
batch_size_test = (1000,1000)


n_steps = 50000
n_hidden, n_layers = 16, 2

n_hidden_2, n_layers_2 = 128, 5

lrate = 1e-4

summary_freq    = 250
test_freq       = 5000
model_save_freq = 10000

seed = 697093

#define NN

c1 = constants.Constants(
            RUN="FBPINN_%s"%(P.w),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            SUBDOMAIN_WS=subdomain_ws,
            BOUNDARY_N=boundary_n,
            Y_N=y_n,
            SEED=seed,
            ACTIVE_SCHEDULER=active_schedulers.AllActiveSchedulerND,
            ACTIVE_SCHEDULER_ARGS=(),
            N_HIDDEN=n_hidden,
            N_LAYERS=n_layers,
            BATCH_SIZE=batch_size,
            LRATE = lrate,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            SUMMARY_FREQ    = summary_freq,
            TEST_FREQ       = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )

c2 = constants.Constants(
            RUN="PINN_%s"%(P.w),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            BOUNDARY_N=boundary_n,
            Y_N=y_n,
            SEED=seed,
            N_HIDDEN=n_hidden_2,
            N_LAYERS=n_layers_2,
            BATCH_SIZE=batch_size,
            LRATE = lrate,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            SUMMARY_FREQ    = summary_freq,
            TEST_FREQ       = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )

c3 = constants.Constants(
            RUN="PINN_FF_%s"%(P.w),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            BOUNDARY_N=boundary_n,
            Y_N=y_n,
            SEED=seed,
            N_HIDDEN=n_hidden_2,
            N_LAYERS=n_layers_2,
            BATCH_SIZE=batch_size,
            LRATE = lrate,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            SUMMARY_FREQ    = summary_freq,
            TEST_FREQ       = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN_FF,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )

c4 = constants.Constants(
            RUN="FBPINN_FF_%s"%(P.w),
            P=P,
            SUBDOMAIN_XS=subdomain_xs_2,
            SUBDOMAIN_WS=subdomain_ws_2,
            BOUNDARY_N=boundary_n,
            Y_N=y_n,
            SEED=seed,
            ACTIVE_SCHEDULER=active_schedulers.AllActiveSchedulerND,
            ACTIVE_SCHEDULER_ARGS=(),
            N_HIDDEN=n_hidden,
            N_LAYERS=n_layers,
            BATCH_SIZE=batch_size,
            LRATE = lrate,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            SUMMARY_FREQ    = summary_freq,
            TEST_FREQ       = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )


# train FBPINN_FCN
run = main.FBPINNTrainer(c1)
run.train()

# train PINN_FCN
run = main.PINNTrainer(c2)
run.train()

# train PINN_FCN_FF
run = main.PINNTrainer(c3)
run.train()

# train FBPINN_FCN_FF
run = main.FBPINNTrainer(c4)
run.train()

# finally, compare runs by plotting saved test losses

fbpinn_loss = np.load("results/models/%s/loss_%.8i.npy"%(c1.RUN, n_steps))
pinn_loss   = np.load("results/models/%s/loss_%.8i.npy"%(c2.RUN, n_steps))
pinn_ff_loss   = np.load("results/models/%s/loss_%.8i.npy"%(c3.RUN, n_steps))
fbpinn_ff_loss = np.load("results/models/%s/loss_%.8i.npy"%(c1.RUN, n_steps))

plt.figure(figsize=(7,5))
plt.plot(fbpinn_loss[:,0], fbpinn_loss[:,3], label=c1.RUN)
plt.plot(pinn_loss[:,0], pinn_loss[:,3], label=c2.RUN)
plt.plot(pinn_ff_loss[:,0], pinn_ff_loss[:,3], label=c3.RUN)
plt.plot(fbpinn_ff_loss[:,0], fbpinn_loss[:,3], label=c4.RUN)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("L2 loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig('loss_comparison_cos_l2')
