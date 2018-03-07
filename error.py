#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def psf_moffat(A,k,l,sigma,beta):
    var =  A/(1+((k*k + l*l)/(sigma*sigma)))**beta
    return var
    
#np.set_printoptions(precision=0)
X, Y, radius, bg, A, beta, sigma = np.loadtxt('data/ocenb-3.csv', usecols=(0,1,4,7,8,11,13), unpack=True)

#Nilai rata-rata
rad_ = round(np.mean(radius))
bg_ = np.mean(bg)
A_ = np.mean(A)
beta_ = np.mean(beta)
sigma_ = np.mean(sigma)

print("R = {} \nBG = {} \nA = {} \nbeta = {} \nsigma = {} \n" .format(rad_,bg_,A_,beta_,sigma_))

drad = np.std(radius)
dbg = np.std(bg)
dA = np.std(A)
dbeta = np.std(beta)
dsigma = np.std(sigma)

k = np.arange(-rad_,rad_+1)
l = np.arange(-rad_,rad_+1)
k,l = np.meshgrid(k,l)

print("dR = {} \ndBG = {} \ndA = {} \ndbeta = {} \ndsigma = {} \n" .format(drad,dbg,dA,dbeta,dsigma))

psf = psf_moffat(A_,k,l,sigma_,beta_)
print(time.clock())

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(k, l, psf, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

"""
#MSE
def mse(data):
    error = []
    K, M, N = data.shape
    
    for i in range(1,K):
        error.append(np.sum((allimg[0]-allimg[i])**2)/(M*N))
    return error

def dmse(data):
    error = []
    K, M, N = data.shape
    
    for i in range(1,K):
        error.append(2*np.sum((allimg[0]-allimg[i]))/(M*N))
    return error
    
def mse2(data):
    error = []
    K, M, N = data.shape
    
    for i in range(1,K):
        error.append(np.sum(abs(allimg[i-1]-allimg[i]))/np.sum(allimg[i-1]))
    return error
"""