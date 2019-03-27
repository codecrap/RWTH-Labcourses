#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:03:27 2019

@author: alex
"""

import matplotlib.pyplot as plt
import numpy as np

# set style for matplotlib
import matplotlib
matplotlib.style.use("../labreport.mplstyle")

# path needed for PraktLib
import sys
sys.path.append("./../../")

# import PraktLib take care of changes in module by manually reloading
import PraktLib as pl
from importlib import reload
pl = reload(pl)

# set strings and choose file
DATAPATH = "./Gwyddion/Gold/"
FILE_PREFIX = ".txt"
FILE_NAME = "TIME_I_profiles"

# set boundaries
#left = 10e-9
#right = 20e-9

# read data
with open(DATAPATH + FILE_NAME + FILE_PREFIX) as data:
		   v01X, v01Y, v02X, v02Y  = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T

# get real boundaries
#rleft, ileft = pl.find_nearest(v1900X, left)
#rright, iright = pl.find_nearest(v1900X, right)

# cut
#_, v19Cut, _ = np.split(v1900Y, [ileft, iright])
#_, v25Cut, _ = np.split(v2500Y, [ileft, iright])

# get maxima
#max19 = np.max(v19Cut)
#max25 = np.max(v25Cut)

# get minima
min01 = np.min(v01Y)
min02 = np.min(v02Y)

# get difference
#diff19 = max19 - min19
#diff25 = max25 - min25

# print
print('RZ = 0.1')
print(min01)
print('RZ = 0.2')
print(min02)

# plot
fig, ax = plt.subplots()
#ax.set_title()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [A]')
#ax.axvline(left, color='r', linestyle='--', linewidth=1)
#ax.axvline(right, color='r', linestyle='--', linewidth=1)
ax.plot(v01X, v01Y, 'b-', label='Rasterzeit 0,1 s')
ax.axhline(min01, color='b', linestyle='--', linewidth=1)
#ax.axhline(min19, color='b', linestyle='--', linewidth=1)
ax.plot(v02X, v02Y, 'k-', label='Rasterzeit 0,2 s')
ax.axhline(min02, color='k', linestyle='--', linewidth=1)
#ax.axhline(min25, color='k', linestyle='--', linewidth=1)
ax.legend(loc='upper right')
fig.savefig('Figures/'+FILE_NAME+'.pdf')
