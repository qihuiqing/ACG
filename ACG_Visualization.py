# coding: utf-8
'''
Created on 2021年1月21日
@author: Chyi
'''
import scipy.stats  as stats
import matplotlib.pyplot as plt
import numpy as np

import math
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        input_size = m.weight.shape[-1]
        s = 1 / math.sqrt(input_size)
        nn.init.trunc_normal_(m.weight, std=s)
        m.bias.data.fill_(0.0)

def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues', ax=None):
    values = np.vstack([mu, tau])
    kernel = stats.gaussian_kde(values)
    if not ax:
        _, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    return ax
