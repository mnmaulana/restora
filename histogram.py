# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:31:48 2018

@author: mnm
"""

import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as fits

# Baca File FITS
data = fits.open('data/ocenv-3.FIT')
image = data[0].data
header = data[0].header

