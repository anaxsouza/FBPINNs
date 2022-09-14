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
import perm_functions as pf

class Darcy_2D(problems._Problem):

    #    Solves the 2D PDE:
    #     d²u    d²u          
    #    ---- + ---- = k*(-8*pi^2*sin(2pi*x)sin(2pi*y))
    #     dx²    dy²     
    #
    #   k = f(x,y)
    #
    # Exact Solution: sin(2pix)sin(2piy)
    #
    # BC:
    # u(x,0) = 0
    # u(0,y) = 0
    
    @property
    def name(self):
        return "Darcy_2D_Masud_%s"%(self)
    
    def __init__(self, run):
        
        # dimensionality of x and y

        self.run = run

        self.d = (2,1)

    def physics_loss(self, x, y, j0, j1, jj0, jj1):

        kx = 1
        ky = 1
        
        k = kx*ky #permeability

        physics = (jj0[:,0] + jj1[:,0]) + (k*(8*np.pi**2)*(torch.sin(2*np.pi*x[:,0])*torch.sin(2*np.pi*x[:,1])))
        
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        
        j0 = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0][:,0:1]
        j1 = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0][:,1:2]

        jj0 = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0][:,0:1]
        jj1 = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0][:,1:2]

        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):

        kx = 1
        ky = 1        
        
        k = kx*ky #permeability

        # Apply u = tanh(x)tanh(x - u(x))*tanh(y)tanh(x - u(y))NN +cos(x)*cos(y) ansatz
        
        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        t1, jt1, jjt1 = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        '''
        bc_y   = torch.cos(np.pi*x[:,0:1])*torch.cos(np.pi*x[:,1:2])
        bc_j0  = -np.pi*torch.sin(np.pi*x[:,0:1])*torch.cos(np.pi*x[:,1:2])
        bc_j1  = -np.pi*torch.cos(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
        bc_jj0 = -(np.pi**2)*torch.cos(np.pi*x[:,0:1])*torch.cos(np.pi*x[:,1:2])
        bc_jj1 = -(np.pi**2)*torch.cos(np.pi*x[:,0:1])*torch.cos(np.pi*x[:,1:2])
        '''
        y_new = k*(t0*t1*y)
        j0_new = k*(jt0*t1*y + t0*t1*j0) 
        j1_new = k*(jt1*t0*y + t1*t0*j1)
        jj0_new = k*(jjt0*t1*y + 2*jt0*t1*j0 + t0*t1*jj0)
        jj1_new = k*(jjt1*t0*y + 2*jt1*t0*j1 + t1*t0*jj1)  

        return y_new, j0_new, j1_new, jj0_new, jj1_new

    def exact_solution(self, x, batch_size):
        
        kx = 1
        ky = 1       
        
        k = kx*ky #permeability

        y = k*(torch.sin(2*np.pi*x[:,0:1])*torch.sin(2*np.pi*x[:,1:2]))
        
        j0 = k*((2*np.pi)*torch.cos(2*np.pi*x[:,0:1])*torch.sin(2*np.pi*x[:,1:2]))
        j1 = k*((2*np.pi)*torch.sin(2*np.pi*x[:,0:1])*torch.cos(2*np.pi*x[:,1:2]))
        
        jj0 = k*((-4*np.pi**2)*torch.sin(2*np.pi*x[:,0:1])*torch.sin(2*np.pi*x[:,1:2]))
        jj1 = k*((-4*np.pi**2)*torch.sin(2*np.pi*x[:,0:1])*torch.sin(2*np.pi*x[:,1:2]))


        return y, j0, j1, jj0, jj1

#domain definition

x_domain_start = 0
y_domain_start = 0

x_domain_end = 1.1
y_domain_end = 1.1

step = .5

run = 'Darcy_Flow'

P = Darcy_2D(run)

#hyper-parameters

width = .8

subdomain_xs = [np.arange(x_domain_start, x_domain_end, step), np.arange(y_domain_start, y_domain_end, step)]
subdomain_ws = constants.get_subdomain_ws(subdomain_xs, width)

boundary_n = (1,)
y_n = (0,1)
batch_size = (50,50)
batch_size_test = (100,100)

n_steps = 50000
n_hidden, n_layers = 16, 2

n_hidden_2, n_layers_2 = 32, 4

lrate = 1e-4

summary_freq    = 250
test_freq       = 2500
model_save_freq = 50000

seed = 697093

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


# train FBPINN_FCN
run = main.FBPINNTrainer(c1)
run.train()

# train PINN_FCN
run = main.PINNTrainer(c2)
run.train()

# train PINN_FCN_FF
run = main.PINNTrainer(c3)
run.train()


# finally, compare runs by plotting saved test losses

fbpinn_loss = np.load("results/models/%s/loss_%.8i.npy"%(c1.RUN, n_steps))
pinn_loss   = np.load("results/models/%s/loss_%.8i.npy"%(c2.RUN, n_steps))
pinn_ff_loss   = np.load("results/models/%s/loss_%.8i.npy"%(c3.RUN, n_steps))

plt.figure(figsize=(7,5))
plt.plot(fbpinn_loss[:,0], fbpinn_loss[:,3], label=c1.RUN)
plt.plot(pinn_loss[:,0], pinn_loss[:,3], label=c2.RUN)
plt.plot(pinn_ff_loss[:,0], pinn_ff_loss[:,3], label=c3.RUN)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("L2 loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig('loss_comparison_darcy_l2')
