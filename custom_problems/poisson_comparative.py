"""
Created on Tue Apr  12 14:28:00 2022

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

class poisson_2D(problems._Problem):

    #    Solves the 2D PDE:
    #    d^2u   d^2u
    #    ---- + ---- = -2*w^2*sin(wx)*sin(wy)
    #    dx^2   dy^2
    #    
    #    Boundary conditions:
    #    u(0,y) = 0
    #    u(x,0) = 0
    #    u(1,y) = sin(wy)
    #    u(x,1) = sin(wx)
    #
    #    Exact Solution: sin(wx)sin(wy)
    
    
    @property
    def name(self):
        return "Poisson_2D_w%s"%(self.w)
    
    def __init__(self, run, w):

        # dimensionality of x and y

        self.run = run

        self.d = (2,1)

        self.w = w
    
    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        
        physics = (jj0[:,0] + jj1[:,0]) + ((2*self.w**2)*torch.sin(self.w*x[:,0])*torch.sin(self.w*x[:,1]))
        
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        jj0, jj1 = jj[:,0:1], jj[:,1:2]

        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        
        # Apply u = tanh(x)tanh(x - u(x))*tanh(y)tanh(x - u(y))NN ansatz
        
        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 10*np.pi, sd)
        t1, jt1, jjt1 = boundary_conditions.sin(x[:,1:2], w, 0, sd)

        y_new = t0*t1*y 
        j0_new = jt0*t1*y + t0*t1*j0  
        j1_new = jt1*t0*y + t1*t0*j1
        jj0_new = jjt0*t1*y + 2*jt0*t1*j0 + t0*t1*jj0
        jj1_new = jjt1*t0*y + 2*jt1*t0*j1 + t1*t0*jj1   

        return y_new, j0_new, j1_new, jj0_new, jj1_new

    def exact_solution(self, x, batch_size):
        
        y = torch.sin(self.w*x[:,0:1])*torch.sin(self.w*x[:,1:2])
        
        j0 = self.w*torch.cos(self.w*x[:,0:1])*torch.sin(self.w*x[:,1:2])
        j1 = self.w*torch.sin(self.w*x[:,0:1])*torch.cos(self.w*x[:,1:2])
        
        jj0 = -(self.w**2)*torch.sin(self.w*x[:,0:1])*torch.sin(self.w*x[:,1:2])
        jj1 = -(self.w**2)*torch.sin(self.w*x[:,0:1])*torch.sin(self.w*x[:,1:2])

        return y, j0, j1, jj0, jj1

run = 'Poisson_2D_Comparative'

w = 10

P = poisson_2D(run, w)

#hyper-parameters

width = .8

subdomain_xs = [np.arange(0, 10.1*np.pi, np.pi/2), np.arange(0, 10.1*np.pi, np.pi/2)]
subdomain_ws = constants.get_subdomain_ws(subdomain_xs, width)

subdomain_xs_2 = [np.arange(0, 10.1*np.pi, np.pi), np.arange(0, 10.1*np.pi, np.pi)]
subdomain_ws_2 = constants.get_subdomain_ws(subdomain_xs_2, width)

subdomain_xs_3 = [np.arange(0, 10.1*np.pi, 2*np.pi), np.arange(0, 10.1*np.pi, 2*np.pi)]
subdomain_ws_3 = constants.get_subdomain_ws(subdomain_xs_3, width)

subdomain_xs_4 = [np.arange(0, 10.1*np.pi, 5*np.pi), np.arange(0, 10.1*np.pi, 5*np.pi)]
subdomain_ws_4 = constants.get_subdomain_ws(subdomain_xs_4, width)

boundary_n = (1,)
y_n = (0, 1)
batch_size = (100,100)
batch_size_test = (200,200)

n_steps = 100000
n_hidden, n_layers = 16, 2

n_hidden_2, n_layers_2 = 64, 8

lrate = 1e-4

summary_freq    = 250
test_freq       = 5000
model_save_freq = 10000

seed = 123

#define NN

c1 = constants.Constants(
            RUN="FBPINN_20_%s"%(P.run),
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
            RUN="FBPINN_10_%s"%(P.run),
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

c3 = constants.Constants(
            RUN="FBPINN_5_%s"%(P.run),
            P=P,
            SUBDOMAIN_XS=subdomain_xs_3,
            SUBDOMAIN_WS=subdomain_ws_3,
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

c4 = constants.Constants(
            RUN="FBPINN_2_%s"%(P.run),
            P=P,
            SUBDOMAIN_XS=subdomain_xs_4,
            SUBDOMAIN_WS=subdomain_ws_4,
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


run = main.FBPINNTrainer(c1)
run.train()

run = main.FBPINNTrainer(c2)
run.train()

run = main.FBPINNTrainer(c3)
run.train()

run = main.FBPINNTrainer(c4)
run.train()

# finally, compare runs by plotting saved test losses

fbpinn_1 = np.load("results/models/%s/loss_%.8i.npy"%(c1.RUN, n_steps))
fbpinn_2 = np.load("results/models/%s/loss_%.8i.npy"%(c2.RUN, n_steps))
fbpinn_3 = np.load("results/models/%s/loss_%.8i.npy"%(c3.RUN, n_steps))
fbpinn_4 = np.load("results/models/%s/loss_%.8i.npy"%(c4.RUN, n_steps))

plt.figure(figsize=(12,10))
plt.plot(fbpinn_1[:,0], fbpinn_1[:,3], label=c1.RUN)
plt.plot(fbpinn_2[:,0], fbpinn_2[:,3], label=c2.RUN)
plt.plot(fbpinn_3[:,0], fbpinn_3[:,3], label=c3.RUN)
plt.plot(fbpinn_4[:,0], fbpinn_4[:,3], label=c4.RUN)

plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("L2 loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig('loss_comparison_poisson_l2')