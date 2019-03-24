#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:54:40 2019

@author: alex
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random

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


# set strings
DATAPATH = "./Gwyddion/HOPG/"
FILE_POSTFIX = ".txt"
vRES = ['300nm']#, '166nm', '955nm']

# set boundaries
vBounds= [[[0.95e-7,1.10e-7], [0.89e-7, 1.03e-7], [0.65e-7,0.79e-7]],
		  [[0,0],[0,0],[0,0]],
		  [[0,0],[0,0],[0,0]]]

### for loop? ###
for i, RES in enumerate(vRES):
	# get data
	with open(DATAPATH + RES + FILE_POSTFIX) as data:
		   vX1, vY1, vX2, vY2, vX3, vY3 = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T
	
	# plot data
	#plt.plot(vX2, vY2, '.')
	#plt.plot(vX2, vY2, '.')
	#plt.plot(vX3, vY3, '.')
	print(pl.find_nearest(vX1,0.95e-7))

