#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:54:40 2019

@author: alex
"""

import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

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
vRES = ['300nm', '166nm', '95nm']

# set boundaries (order: 300, 166, 95 and 1, 2, 3)
vBounds= [[[0.95e-7,1.10e-7], [0.89e-7, 1.03e-7], [0.65e-7,0.79e-7]],
		  [[2.2e-8,3.2e-8], [1.8e-8,2.8e-8], [1.9e-8,2.7e-8]],
		  [[2.9e-8,3.9e-8], [2.4e-8,3.5e-8], [2.2e-8,3.4e-8]]]

vHeights = [[],[],[]]
vMeans = []

for i, RES in enumerate(vRES):
	# get data
	with open(DATAPATH + RES + FILE_POSTFIX) as data:
		   vX1, vY1, vX2, vY2, vX3, vY3 = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T
	
	# make data loopable
	vX = [vX1, vX2, vX3]
	vY = [vY1, vY2, vY3]
	
	# plot data
	#plt.plot(vX1, vY1, '.')
	#plt.plot(vX2, vY2, '.')
	#plt.plot(vX3, vY3, '.')
	
	for j in range(3):
		# get rid on nans
		vX[j] = vX[j][~np.isnan(vX[j])]
		vY[j] = vY[j][~np.isnan(vY[j])]
		
		# find boundary indexes
		_, idx1 = pl.find_nearest(vX[j], vBounds[i][j][0])
		_, idx2 = pl.find_nearest(vX[j], vBounds[i][j][1])
		vIdx = [idx1, idx2]
		
		# split array
		vTop, _, vBottom = np.split(vY1, vIdx)
		
		# calculate means
		top = ufloat(np.mean(vTop), np.std(vTop, ddof=1))
		bottom = ufloat(np.mean(vBottom), np.std(vBottom, ddof=1))
		
		# calculate height
		vHeights[j] += [top-bottom]
		
	# calculate weighted mean
	vH = abs(unp.nominal_values(vHeights[j]))
	vHErr = unp.std_devs(vHeights[j])
	mean, sigma = pl.weightedMean(vH, vHErr)
	vMeans += [ufloat(mean, sigma)]

print(vHeights)	 
print(vMeans)
