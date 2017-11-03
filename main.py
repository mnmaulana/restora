#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:40:51 2017

@author: mnm
"""


import numpy as np 
import matplotlib.pyplot as pl
from astropy.io import fits
#import fitsio as fits
import time
from scipy.signal import convolve

np.seterr(divide='ignore')

def richardson_lucy(image, psf, iterations=50):
    
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    bg = np.min(image)
    im_deconv = bg * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        relative_blur = image / convolve(im_deconv, psf, 'same')
        im_deconv *= convolve(relative_blur, psf_mirror, 'same')
        #im_deconv[im_deconv == np.inf] = 0
        #im_deconv = np.nan_to_num(im_deconv)
    
    return im_deconv


#def psf_gauss(A,k,l,sigma): 
#    return A*np.exp(-((k-x)**2 + (l-y)**2)/(2*sigma**2))
    
def psf_moffat(A,k,l,sigma,beta,B=0):
    return (A/(1+((k*k + l*l)/(sigma*sigma)))**beta) + B


data = fits.open('')

image = data[0].data
#img = fits.read('/home/mnm/Documents/Course/finalproject/Data_Evan_Schmidt_2013/Omega_centauri_Schmidt/ocen-1.FIT')
#img = 10*np.ones([99,99])
#img[50,50] = 1000

r = 7

k = np.arange(-r,r+1)
l = np.arange(-r,r+1)
k,l = np.meshgrid(k,l)

A = 10
#B = np.min(img)
sigma = 5
beta = 5

model = psf_moffat(A,k,l,sigma,beta)
#print(model.shape)
#img_conv = convolve(img,model,'same')
deconv = richardson_lucy(img,model,10)

pl.figure()
pl.subplot(221)
pl.imshow(img,cmap='gray',label="ideal image")
pl.subplot(222)
pl.imshow(model,cmap='gray',label="psf model")
pl.subplot(223)
#pl.imshow(img_conv,cmap='gray',label="degradage image")
pl.subplot(224)
pl.imshow(deconv,cmap='gray',label="restored image")


"""
x = np.median(k)
y = np.median(l)
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

#pl.imshow(img, cmap='gray')
print(time.clock())
pl.show()
data.close()