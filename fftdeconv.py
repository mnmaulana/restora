# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:17:42 2018

@author: mnm
"""

import numpy as np 
import matplotlib.pyplot as pl
from astropy.io import fits
from scipy.signal import fftconvolve
import time


def richardson_lucy(image, psf, iterations=50):
    
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    BG = np.mean(image)
    im_deconv = BG * np.ones(image.shape)
    print("BG OK! bg = {}" .format(BG))
    psf_mirror = psf[::-1, ::-1]
    M, N = image.shape
    Z_XY = np.zeros((iterations+1,M,N)) 
    Z_XY[0] = image    
    
    for _ in range(iterations):
        in_t = time.time()
        relative_blur = image / fftconvolve(im_deconv, psf, 'same')
        im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')
        print("Progress {} / {} in {} s" .format(_+1,iterations,time.time()-in_t))
        
        
        Z_XY[_+1] = im_deconv
#        mserr =  np.sum((Z_XY[_]-im_deconv)**2)/(M*N)
        
#        if mserr <= 0.02*BG:
#            print("Convergence... \nRestoration is Stopped in {} iterations".format(_))
#            break
        
        #if _ > 19:
        #SIMPAN HASIL RESTORASI KE DALAM FILE FITS        
        in_t = time.time()        
        filename = "Restored-{}.fits".format(_+1)
        fitsimg = im_deconv.astype(np.uint16) 
        hdu = fits.PrimaryHDU(fitsimg, header)
        hdu.writeto(filename)
        print("Simpan Citra {} / {} in {} s" .format(_+1,iterations,time.time()-in_t))
        
    return im_deconv, Z_XY

    
def psf_moffat(A,k,l,sigma,beta):
    var =  A/(1+((k*k + l*l)/(sigma*sigma)))**beta
    return var
    
    
t_in = time.time()

# Baca File FITS
data = fits.open('data/ocenb-4.fits')
image = data[0].data
header = data[0].header
print("FITS OK!")

# Baca File PSF
radius, bg, A, beta, sigma = np.loadtxt('data/ocenb-4.csv', usecols=(4,7,8,11,13), unpack=True)

#Nilai rata-rata
rad_ = np.mean(radius)
bg_ = np.mean(bg)
A_ = np.mean(A)
beta_ = np.mean(beta)
sigma_ = np.mean(sigma)

rad_ = round(rad_)
k = np.arange(-rad_,rad_+1)
l = np.arange(-rad_,rad_+1)
k,l = np.meshgrid(k,l)

print(rad_,A_,beta_,sigma_)

psf = psf_moffat(A_,k,l,sigma_,beta_)
print("PSF OK!")
print("restoration started")
deconv, allimg = richardson_lucy(image, psf ,25)
allimg = np.array(allimg)
#error = mse(allimg)
#derr = dmse(allimg)
#error2 = mse2(allimg)

print("Restoration done! ")
#print("MSE = ", Emse)


data.close()