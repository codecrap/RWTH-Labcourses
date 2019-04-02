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

vCounts = np.genfromtxt(DATAPATH + "HOM-Dip_noFilter" + FILE_POSTFIX,
							dtype=float, delimiter='\t', skip_header=3, usecols=(3) )

fig,ax = plt.subplots()
ax.plot(np.arange(2,len(vCounts)*0.5+2,0.5),vCounts,'ro-')

# >>> myList = [["A", 20, False], ["B", 1, False], ["C", 8, False]]
# >>> smallest = min(myList, key=lambda L: L[1])
# >>> smallest
# ['B', 1, False]