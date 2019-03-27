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
FILE_NAME = "GAIN_I_forward_profiles"

# set boundaries
left = 10e-9
right = 20e-9

# read data
with open(DATAPATH + FILE_NAME + FILE_PREFIX) as data:
		   v1900X, v1900Y, v2500X, v2500Y, _, _ = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T

# get real boundaries
rleft, ileft = pl.find_nearest(v1900X, left)
rright, iright = pl.find_nearest(v1900X, right)

# cut
_, v19Cut, _ = np.split(v1900Y, [ileft, iright])
_, v25Cut, _ = np.split(v2500Y, [ileft, iright])

# get maxima
max19 = np.max(v19Cut)
max25 = np.max(v25Cut)

# get minima
min19 = np.min(v19Cut)
min25 = np.min(v25Cut)

# get difference
diff19 = max19 - min19
diff25 = max25 - min25

# print
print('Gain = 1900')
print(diff19)
print('Gain = 2500')
print(diff25)

# plot
fig, ax = plt.subplots()
#ax.set_title()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [A]')
ax.axvline(left, color='r', linestyle='--', linewidth=1)
ax.axvline(right, color='r', linestyle='--', linewidth=1)
ax.plot(v1900X, v1900Y, 'b-', label='Gain 1900')
ax.axhline(max19, color='b', linestyle='--', linewidth=1)
ax.axhline(min19, color='b', linestyle='--', linewidth=1)
ax.plot(v2500X, v2500Y, 'k-', label='Gain 2500')
ax.axhline(max25, color='k', linestyle='--', linewidth=1)
ax.axhline(min25, color='k', linestyle='--', linewidth=1)
ax.legend(loc='upper right')
fig.savefig('Figures/'+FILE_NAME+'.pdf')
