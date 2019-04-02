#-*- coding: utf-8 -*-
#
#@statistics_check.py:
#@author: Olexiy Fedorets
#@date: Thu 14.03.2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


DATAPATH = "./Data/Day1/Poisson_statistics_6.1.1/"
FILE_POSTFIX = ".quCNTPlot"
vCURRENT = ['30','37','45']


for i,current in enumerate(vCURRENT):
	mData = np.genfromtxt(DATAPATH + current + 'mA' + FILE_POSTFIX,
							dtype=float, delimiter='\t', skip_header=1, usecols=(1,2))
	mData = mData[np.nonzero(mData)[0],:]
	mData = mData[-11:-1]

	print(current + 'mA')
	print("Sample & {0:.0f} \pm {2:.0f} & {1:.0f} \pm {3:.0f} "
		  .format(*np.mean(mData,axis=0), *np.std(mData,axis=0,ddof=1) ) )
	print("Poisson & {0:.0f} \pm {2:.0f} & {1:.0f} \pm {3:.0f} "
		  .format(*np.mean(mData,axis=0), *np.sqrt(np.mean(mData,axis=0)) ) )

	# print(" {.1f} +- {.1f} ".format(np.mean(mData,axis=0), np.sqrt(np.mean(mData,axis=0)) ) )
	# print("Sample  ", np.mean(mData,axis=0), np.std(mData,axis=0,ddof=1) )
	# print("Poisson ", np.mean(mData,axis=0), np.sqrt(np.mean(mData,axis=0)) )
	
