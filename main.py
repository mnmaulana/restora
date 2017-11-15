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

#np.seterr(divide='ignore')

def richardson_lucy(image, psf, iterations=50):
    
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    #bg = np.min(image)
    im_deconv = bg * np.ones(image.shape)
    print("BG OK! bg = {}" .format(bg))
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        in_t = time.time()
        relative_blur = image / convolve(im_deconv, psf, 'same')
        im_deconv *= convolve(relative_blur, psf_mirror, 'same')
        #im_deconv[im_deconv == np.inf] = 0
        #im_deconv = np.nan_to_num(im_deconv)
        print("Progress {} / {} in {} s" .format(_+1,iterations,time.time()-in_t))
    
    return im_deconv


#def psf_gauss(A,k,l,sigma): 
#    return A*np.exp(-((k-x)**2 + (l-y)**2)/(2*sigma**2))
    
def psf_moffat(A,k,l,sigma,beta):
    return (A/(1+((k*k + l*l)/(sigma*sigma)))**beta)

t_in = time.time()

data = fits.open('ocenb-4.fits')

image = data[0].data
header = data[0].header

#image = 10*np.ones([99,99])
#image[50,50] = 1000
#image[20,15] = 500
#image[80,65] = 700

print("FITS OK!")
#img = fits.read('/home/mnm/Documents/Course/finalproject/Data_Evan_Schmidt_2013/Omega_centauri_Schmidt/ocenb-4.FIT')

r = 12

k = np.arange(-r,r+1)
l = np.arange(-r,r+1)
k,l = np.meshgrid(k,l)

A = 1088.92
bg = 1811.56
sigma = 2.71
beta = 6.37

psf = psf_moffat(A,k,l,sigma,beta)
print("PSF OK!")
#print(model.shape)
#img_conv = convolve(img,model,'same')
print("Entering restoration")
deconv = richardson_lucy(image, psf ,20)
print("Restoration done! \nPlotting....")
'''
pl.figure()
pl.subplot(221)
pl.imshow(img,cmap='gray',label="ideal image")
pl.subplot(222)
pl.imshow(model,cmap='gray',label="psf model")
pl.subplot(223)
#pl.imshow(img_conv,cmap='gray',label="degradage image")
pl.subplot(224)
pl.imshow(deconv,cmap='gray',label="restored image")
'''


#x = np.median(k)
#y = np.median(l)
#moffat = psf_moffat(A,k,l,sigma,beta)
print("CPU time = ", time.time() - t_in)

pl.figure(0)
pl.imshow(psf, cmap='gray')
pl.figure(1)
pl.imshow(image, cmap='gray')
pl.figure(2)
pl.imshow(deconv, cmap='gray')

#gauss = psf_gauss(A,k,l,sigma)
"""
pl.figure()
pl.subplot(121)
pl.title('PSF Moffat')
pl.imshow(moffat, cmap='gray')
pl.subplot(122)
pl.title('PSF Gauss')
pl.imshow(gauss, cmap='gray')
"""

#pl.imshow(img, cmap='gray')
#pl.show()
data.close()