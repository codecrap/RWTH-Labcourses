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


DATAPATH = "./Data/"
FILE_POSTFIX = ".quCNTPlot"

mData = np.genfromtxt(DATAPATH + "2016-08-31-10-12-14" + FILE_POSTFIX,
							dtype=float, delimiter='\t', skip_header=1, usecols=(1,2))

# print("Sample %.1f +- %.1f " % (np.mean(mData,axis=0), np.std(mData,axis=0,ddof=1)) )
# print("Poisson %.1f +- %.1f " % (np.mean(mData,axis=0), np.sqrt(np.mean(mData,axis=0))) )
print("Sample  ", np.mean(mData,axis=0), np.std(mData,axis=0,ddof=1) )
print("Poisson ", np.mean(mData,axis=0), np.sqrt(np.mean(mData,axis=0)) )