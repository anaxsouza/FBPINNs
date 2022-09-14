"""
Created on Tue Sep  07 10:31:00 2022

@author: anaxsouza
"""

import torch
import numpy as np


def sin(x, w, mu):

    k = torch.sin(w*x - mu)

    return k


def cos(x, w, mu):

    k = torch.cos(w*x - mu)

    return k

def tan(x, w, mu):

    k = torch.tan(w*x -mu)

    return k

def sinh(x, w, mu):

    k = torch.sinh(w*x - mu)

    return k

def cosh(x, w, mu):

    k = torch.cosh(w*x - mu)

    return k

def tanh(x, w, mu):

    k = torch.tanh(w*x - mu)

    return k

