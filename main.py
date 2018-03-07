#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np 
from astropy.io import fits
from scipy.signal import fftconvolve
import time


# Fungsi utama untuk melakukan restorasi
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
        print("Iterasi ke - {} / {} dalam {} detik" .format(_+1,iterations,time.time()-in_t))
        
        # hasil restorasi ditambahkan kedalam array baru. Digunakan untuk menghitung MSE dan PSNR
        Z_XY[_+1] = im_deconv 
    return im_deconv, Z_XY


# Fungsi untuk membuat PSF    
def psf_moffat(A,k,l,sigma,beta):
    var =  A/(1+((k*k + l*l)/(sigma*sigma)))**beta
    return var


# Fungsi untuk menghitung error 
def mse(data):
    error = []
    K, M, N = data.shape
    
    for i in range(1,K):
        error.append(np.sum((allimg[0]-allimg[i])**2)/(M*N))
    error = np.array(error)
    return error
    
def dmse(data):
    error = []
    K, M, N = data.shape
    
    for i in range(1,K):
        error.append(np.sum((allimg[i-1]-allimg[i])**2)/(M*N))
    error = np.array(error)
    return error
    
def psnr(data):
    error = 20*np.log10(65535)-10*np.log10(data)
    return error
    
t_in = time.time()

print("#####################################")
print("# Program Restorasi Citra Astronomi #")
print("#####################################")

# Input user
fitsname = input("Citra masukan: ")
psfname = input("Parameter PSF: ")
niter = int(input("Jumlah iterasi: "))


# Baca File FITS
data = fits.open(fitsname)
# Memisahkan data citra dan header
image = data[0].data
header = data[0].header
print("FITS OK!")

# Baca File PSF
radius, bg, A, beta, sigma = np.loadtxt(psfname, usecols=(4,7,8,11,13), unpack=True)

# Nilai rata-rata
rad_ = np.mean(radius)
bg_ = np.mean(bg)
A_ = np.mean(A)
beta_ = np.mean(beta)
sigma_ = np.mean(sigma)

rad_ = round(rad_)
k = np.arange(-rad_,rad_+1)
l = np.arange(-rad_,rad_+1)
k,l = np.meshgrid(k,l)

print("Radius = {} \nAmplitudo = {} \nBeta = {} \nFWHM = {}".format(rad_,A_,beta_,sigma_))

psf = psf_moffat(A_,k,l,sigma_,beta_)
print("PSF OK!")
print("Restorasi dimulai")
deconv, allimg = richardson_lucy(image, psf ,niter)
allimg = np.array(allimg)
MSE = mse(allimg)
DMSE = dmse(allimg)
PSNR = psnr(MSE)
DPSNR = psnr(DMSE)

print("Simpan citra keluaran")        
np.savetxt("MSE.txt",MSE,fmt="%1.4e", delimiter="\t")
np.savetxt("DMSE.txt",DMSE,fmt="%1.4e", delimiter="\t")
np.savetxt("PSNR.txt",PSNR,fmt="%1.4e", delimiter="\t")
np.savetxt("DPSNR.txt",DPSNR,fmt="%1.4e", delimiter="\t")

filename = "Restored-{}.fits".format(niter)
fitsimg = deconv.astype(np.uint16) 
hdu = fits.PrimaryHDU(fitsimg, header)
hdu.writeto(filename)

print("Restorasi selesai")

data.close()