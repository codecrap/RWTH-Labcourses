#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:01:51 2019

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
vRES = ['3nm', '2,5nm', '2nm']

# set errors (in nano)
vErr = [[0.08,0.06], [0.09,0.06], [0.08,0.05]]
vErr = np.sqrt(2) * np.array(vErr)

# set triangles
g = 246e-3
vBX = np.array([[3,4,4], [3,3,3], [3,1,2]]) * g
vBY = np.array([[1,1,1], [1,1,1], [1,0,1]]) * g
vCX = np.array([[5,7,7], [5,5,5], [5,2,3]]) * g
vCY = np.array([[11,11,11], [13,13,13], [7,2,7]]) * g

# set angle
phi = pl.degToSr(60)

for i, RES in enumerate(vRES):
	# get data
	with open(DATAPATH + RES + FILE_POSTFIX) as data:
		vX, vY, _, _, _ = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=1).T
	
	# fix 2nm.txt
	if i==2:
		for j in range(3):
			vX[j], vX[j+3] = vX[j+3], vX[j]
			vY[j], vY[j+3] = vY[j+3], vY[j]
	
	# cut
	vX = np.array(vX[:3])
	vY = np.array(vY[3:])
	
	# pico to nano
	vX = np.abs(vX * 1e-3)
	vY = np.abs(vY * 1e-3)
	
	# calc a_theo
	vXTheo = np.array([np.sqrt(vBX[i][k]**2 + vCX[i][k]**2 + vBX[i][k]*vCX[i][k]*np.cos(phi)) for k in range(3)])
	vYTheo = np.array([np.sqrt(vBY[i][k]**2 + vCY[i][k]**2 + vBY[i][k]*vCY[i][k]*np.cos(phi)) for k in range(3)])
	
#	# print intermediate results
#	print(RES)
#	print('Theo:')
#	print(vXTheo)
#	print(vYTheo)
#	print('Exp:')
#	print(vX)
#	print(vY)
	
	# calculate calibration constants
	vKX = np.array([vXTheo[n]/vX[n] for n in range(3)])
	vKXErr = np.array([vErr[i][0] * vXTheo[n]/vX[n]**2 for n in range(3)])

	vKY = np.array([vYTheo[n]/vY[n] for n in range(3)])
	vKYErr = np.array([vErr[i][1] * vYTheo[n]/vY[n]**2 for n in range(3)])
	
	# calculate means
	kx = pl.weightedMean(vKX, vKXErr)
	ky = pl.weightedMean(vKY, vKYErr)
	
	# print end results
	print(RES)
	print('Cal. const.:')
	print(kx)
	#print(vKX)
	print(ky)
	#print(vKY)
	
	
	