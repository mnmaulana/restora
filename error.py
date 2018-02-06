#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np


#np.set_printoptions(precision=0)
X, Y, radius, bg, A, beta, sigma = np.loadtxt('data/ocenb-4.csv', usecols=(0,1,4,7,8,11,13), unpack=True)

#Nilai rata-rata
rad_ = np.mean(radius)
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

print("dR = {} \ndBG = {} \ndA = {} \ndbeta = {} \ndsigma = {} \n" .format(drad,dbg,dA,dbeta,dsigma))

print(time.clock())

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