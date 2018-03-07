# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:30:32 2018

@author: mnm
"""

import numpy as np

detected = np.loadtxt('ocenv-3.FIT.coo.1')
x = [len(detected)]
for i in range(1,26):
    file="Restored-ocenv-3-{}.fits.coo.1".format(i)
    detected = np.loadtxt(file,usecols={1})
    x.append(len(detected))

print(x)