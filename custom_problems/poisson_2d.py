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
    #    ---- + ---- = -2*sin(x)*sin(y)
    #    dx^2   dy^2
    #    
    #    Boundary conditions:
    #    u(0,y) = 0
    #    u(x,0) = 0
    #    u(1,y) = sin(1)*sin(y)
    #    u(x,1) = sin(x)*sin(1)
    #
    #    Exact Solution: sin(x)sin(y)
    
    
    @property
    def name(self):
        return "poisson_2D_%s"%(self)
    
    def __init__(self, run):

        # dimensionality of x and y

        self.run = run

        self.d = (2,1)
    
    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        
        physics = (jj0[:,0] + jj1[:,0]) + (2*torch.sin(x[:,0])*torch.sin(x[:,1]))
        
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        jj0, jj1 = jj[:,0:1], jj[:,1:2]

        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        
        # Apply u = tanh(x)tanh(x - u(x))*tanh(y)tanh(x - u(y))NN ansatz
        
        #t0, jt0, jjt0 = boundary_conditions.cos(x[:,0:1], 1, np.pi/2, sd)
        #t1, jt1, jjt1 = boundary_conditions.cos(x[:,1:2], 1, np.pi/2, sd)
        
        t0 = torch.sin(x[:,0:1])
        jt0 = torch.cos(x[:,0:1])
        jjt0 = -torch.sin(x[:,0:1])

        t1 = torch.sin(x[:,1:2])
        jt1 = torch.cos(x[:,1:2])
        jjt1 = -torch.sin(x[:,1:2])
  
        y_new = t0*t1*y 
        j0_new = jt0*t1*y + t0*t1*j0  
        j1_new = jt1*t0*y + t0*t1*j1 
        jj0_new = jjt0*t1*y + 2*jt0*t1*j0 + t0*t1*jj0  
        jj1_new = jjt1*t1*y + 2*jt0*t1*j1 + t0*t1*jj1 

        return y_new, j0_new, j1_new, jj0_new, jj1_new

    def exact_solution(self, x, batch_size):
        
        y = torch.sin(x[:,0:1])*torch.sin(x[:,1:2])
        
        j0 = torch.cos(x[:,0:1])*torch.sin(x[:,1:2])
        j1 = torch.sin(x[:,0:1])*torch.cos(x[:,1:2])
        
        jj0 = -torch.sin(x[:,0:1])*torch.sin(x[:,1:2])
        jj1 = -torch.sin(x[:,0:1])*torch.sin(x[:,1:2])

        return y, j0, j1, jj0, jj1

run = 'Poisson_1'

P = poisson_2D(run)

#hyper-parameters

width = .8

subdomain_xs = [np.arange(0, 2.1*np.pi, np.pi), np.arange(0, 2.1*np.pi, np.pi)]
subdomain_ws = constants.get_subdomain_ws(subdomain_xs, width)

subdomain_xs_2 = [np.arange(0, 12.1*np.pi, 4*np.pi), np.arange(0, 12.1*np.pi, 4*np.pi)]
subdomain_ws_2 = constants.get_subdomain_ws(subdomain_xs_2, width)

boundary_n = (1,)
y_n = (0, 1)
batch_size = (50,50)
batch_size_test = (100,100)

n_steps = 100000
n_hidden, n_layers = 16, 2

n_hidden_2, n_layers_2 = 32, 4

lrate = 1e-4

summary_freq    = 250
test_freq       = 250
model_save_freq = 50000

seed = 123

#define NN

c1 = constants.Constants(
            RUN="FBPINN_%s"%(P.run),
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
            RUN="PINN_%s"%(P.run),
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
            RUN="PINN_FF_%s"%(P.run),
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
            RUN="FBPINN_FF_%s"%(P.run),
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
            MODEL = models.FCN_FF,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )
'''
# train FBPINN_FCN
run = main.FBPINNTrainer(c1)
run.train()
'''
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
fbpinn_ff_loss = np.load("results/models/%s/loss_%.8i.npy"%(c4.RUN, n_steps))

plt.figure(figsize=(12,10))
plt.plot(fbpinn_loss[:,0], fbpinn_loss[:,3], label=c1.RUN)
plt.plot(pinn_loss[:,0], pinn_loss[:,3], label=c2.RUN)
plt.plot(pinn_ff_loss[:,0], pinn_ff_loss[:,3], label=c3.RUN)
plt.plot(fbpinn_ff_loss[:,0], fbpinn_loss[:,3], label=c4.RUN)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("L2 loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig('loss_comparison_poisson_1_l2')