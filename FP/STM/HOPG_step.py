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

### for loop? ###
for i, RES in enumerate(vRES):
	# get data
	with open(DATAPATH + RES + FILE_POSTFIX) as data:
		   vX1, vY1, vX2, vY2, vX3, vY3 = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T
	
	# plot data
	plt.plot(vX1, vY1)

