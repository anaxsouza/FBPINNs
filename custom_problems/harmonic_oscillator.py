#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:32:55 2021
Modified on Tue Apr  12 14:28:00 2022

@author: bmoseley
@modified by: anaxsouza
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


class HarmonicOscillator1D(problems._Problem):
    
    """Solves the 1D ODE:
          d^2 u      du
        m ----- + mu -- + kx = 0
          dx^2       dx
                
        with the boundary conditions:
        u (0) = 1
        u'(0) = 0
        
    """
    
    @property
    def name(self):
        return "HarmonicOscillator_%s-%s"%(self.delta, self.w0)# can be used for automatic labeling of runs
    
    def __init__(self, delta, w0):
                
        self.d = (1,1)# dimensionality of input variables and solution (d, d_u)
        
        # we also store some useful problem variables too        
        self.delta, self.w0 = delta, w0
        self.mu, self.k = 2*delta, w0**2# invert for mu, k given delta, w0 and fixing m=1 (without loss of generality)
        
    def physics_loss(self, x, y, j, jj):
        
        physics = jj + self.mu*j + self.k*y

        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):

        # for this problem we require j = du/dx and jj = d^2u/dx^2        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        
        return y, j, jj
    
    def boundary_condition(self, x, y, j, jj, sd):
        
        # for this problem the boundary conditions are: u(0) = 1, u'(0) = 0. To satisy these constraints, we use
        # the following constrained solution ansatz:
        # u = 1 + tanh^2((x-0)/sd)*NN
        
        t2, jt2, jjt2 = boundary_conditions.tanh2_2(x,0,sd)# use the helper boundary_conditions module to get gradients of tanh^2
        y_new = t2*y + 1
        j_new = jt2*y + t2*j# apply product rule
        jj_new = jjt2*y + 2*jt2*j + t2*jj# apply product rule
        
        return y_new, j_new, jj_new

    def exact_solution(self, x, batch_size):
        
        # we calculate the exact solution as derived in https://beltoforion.de/en/harmonic_oscillator/
        # we assume the boundary conditions are u(0) = 1, u'(0) = 0
        
        d,w0 = self.delta, self.w0
        if d < w0: # underdamped case
            w = np.sqrt(w0**2-d**2)
            phi = np.arctan(-d/w)
            A = 1/(2*np.cos(phi))
            cos = torch.cos(phi+w*x)
            sin = torch.sin(phi+w*x)
            exp = torch.exp(-d*x)
            y  = exp*2*A*cos
            j  = exp*2*A*(-d*cos-w*sin)
            jj = exp*2*A*((d**2-w**2)*cos+2*d*w*sin)
        elif d == w0: # critically damped case
            A,B = 1,d
            exp = torch.exp(-d*x)
            y = exp*(A+x*B)
            j = -d*y + B*exp
            jj = (d**2)*y - 2*d*B*exp
        else: # overdamped case
            a = np.sqrt(d**2-w0**2)
            d1, d2 = a-d, -a-d
            A = -d2/(2*a)
            B =  d1/(2*a)
            exp1 = torch.exp(d1*x)
            exp2 = torch.exp(d2*x)
            y = A*exp1 + B*exp2
            j = d1*A*exp1 + d2*B*exp2
            jj = (d1**2)*A*exp1 + (d2**2)*B*exp2
            
        return y, j, jj


#hyper-parameters

width = .8

subdomain_xs = [np.arange(0, 11, 1)]
subdomain_ws = constants.get_subdomain_ws(subdomain_xs, width)

y_n = (-1,1)
batch_size = (200,)
batch_size_test = (400,)

n_steps = 10000
n_hidden, n_layers = 16, 2

lrate = 1e-3

summary_freq    = 250
test_freq       = 5000
model_save_freq = 10000

#no damping
#delta = 0
#w0 = 2

#underdamped
delta = 0.2
w0 = 5

#overdamped
#delta = 3
#w0 = 2

#critically damped
#delta = 2
#w0 = 2


P = HarmonicOscillator1D(delta, w0)

c1 = constants.Constants(
            RUN="FBPINN_%s"%(P.name),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            SUBDOMAIN_WS=subdomain_ws,
            BOUNDARY_N=(1/P.w0,),
            Y_N=y_n,
            ACTIVE_SCHEDULER=active_schedulers.AllActiveSchedulerND,
            ACTIVE_SCHEDULER_ARGS=(),
            N_HIDDEN=n_hidden,
            N_LAYERS=n_layers,
            BATCH_SIZE=batch_size,
            BATCH_SIZE_TEST = batch_size_test,
            N_STEPS=n_steps,
            LRATE = lrate,
            SUMMARY_FREQ = summary_freq,
            TEST_FREQ = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )


c2 = constants.Constants(
            RUN="PINN_%s"%(P.name),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            BOUNDARY_N=(1/P.w0,),
            Y_N=y_n,
            N_HIDDEN=32,
            N_LAYERS=3,
            BATCH_SIZE=batch_size,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            LRATE = lrate,
            SUMMARY_FREQ = summary_freq,
            TEST_FREQ = test_freq,
            MODEL_SAVE_FREQ = model_save_freq,
            MODEL = models.FCN,
            PLOT_LIMS=(2, False),
            SHOW_FIGURES = False,
            SAVE_FIGURES = True,
            CLEAR_OUTPUT=True,
            )

c3 = constants.Constants(
            RUN="PINN_FF_%s"%(P.name),
            P=P,
            SUBDOMAIN_XS=subdomain_xs,
            BOUNDARY_N=(1/P.w0,),
            Y_N=y_n,
            N_HIDDEN=32,
            N_LAYERS=3,
            BATCH_SIZE=batch_size,
            N_STEPS=n_steps,
            BATCH_SIZE_TEST=batch_size_test,
            LRATE = lrate,
            SUMMARY_FREQ = summary_freq,
            TEST_FREQ = test_freq,
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
pinn_loss = np.load("results/models/%s/loss_%.8i.npy"%(c2.RUN, n_steps))
pinn_ff_loss   = np.load("results/models/%s/loss_%.8i.npy"%(c3.RUN, n_steps))


plt.figure(figsize=(10,8))
plt.plot(fbpinn_loss[:,0], fbpinn_loss[:,3], label=c1.RUN)
plt.plot(pinn_loss[:,0], pinn_loss[:,3], label=c2.RUN)
plt.plot(pinn_ff_loss[:,0], pinn_ff_loss[:,3], label=c3.RUN)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("L2 loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig('loss_comparison_harmonic_oscillator')