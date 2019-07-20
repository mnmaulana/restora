#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:40:51 2017

@author: mnm
"""


import numpy as np 
import matplotlib.pyplot as pl
import fitsio as fits
from scipy.signal import convolve

def richardson_lucy(image, psf, iterations=50):
    
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve(relative_blur, psf_mirror, 'same')

    return im_deconv


def psf_gauss(A,k,l,sigma): 
    return A*np.exp(-((k-x)**2 + (l-y)**2)/(2*sigma**2))
    


def psf_moffat(A,k,l,sigma,beta):
    return A/(1+(((k-x)**2 + (l-y)**2)/sigma**2))**beta


img = fits.read('/home/mnm/Documents/Course/finalproject/Data_Evan_Schmidt_2013/Omega_centauri_Schmidt/ocen-1.FIT')


"""
k = np.arange(20)
l = np.arange(20)
x = np.median(k)
y = np.median(l)

k,l = np.meshgrid(k,l)

A = 10
sigma = 10
beta = 5

moffat = psf_moffat(A,k,l,sigma,beta)
gauss = psf_gauss(A,k,l,sigma)


pl.figure()
pl.subplot(121)
pl.title('PSF Moffat')
pl.imshow(moffat, cmap='gray')
pl.subplot(122)
pl.title('PSF Gauss')
pl.imshow(gauss, cmap='gray')
"""

pl.imshow(img, cmap='gray')
pl.show()
