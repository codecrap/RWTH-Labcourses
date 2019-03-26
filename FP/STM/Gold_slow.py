#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:32:04 2019

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
FILE_NAME = "GAIN_opt_profile"
#FILE_NAME = "GAIN_slow_profile"
#FILE_NAME = "TIME_01_profile"
#FILE_NAME = "TIME_005_profile"

# read data
with open(DATAPATH + FILE_NAME + FILE_PREFIX) as data:
		   vForwardX, vForwardY, vBackwardX, vBackwardY = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T

# calculate difference between minima
minF = vForwardX[np.argmin(vForwardY)]
minB = vBackwardX[np.argmin(vBackwardY)]
diff = np.abs(minF - minB)
print(diff)

# plot
fig, ax = plt.subplots()
#ax.set_title()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.plot(vForwardX, vForwardY, 'b-', label='forward')
ax.axvline(minF, color='b', linestyle='--', linewidth=1)
ax.plot(vBackwardX, vBackwardY, 'k-', label='backward')
ax.axvline(minB, color='k', linestyle='--', linewidth=1)
ax.legend(loc='upper right')
fig.savefig('Figures/'+FILE_NAME+'.pdf')

