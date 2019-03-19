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
FILE_POSTFIX = ".txt"

vDet0,vDet1,vCoincidence = np.genfromtxt(DATAPATH + "entangled_det0-0deg_det1curve" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, usecols=(1,2,3), unpack=True)

fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vDet1.size),vDet0,'ro-',
		np.linspace(0,280,vDet1.size),vDet1,'bo-',
		np.linspace(0,280,vDet1.size),vCoincidence,'go-')