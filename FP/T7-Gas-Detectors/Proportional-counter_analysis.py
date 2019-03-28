#-*- coding: utf-8 -*-
#
#@proportional_curve_plot.py:
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
vSOURCES = ["AM","C","EMPTY"]


for i,source in enumerate(vSOURCES):
	vU, vCounts = np.genfromtxt(DATAPATH + "Proportional_characteristic_curve_pulse_heights_" + source + FILE_POSTFIX,
										dtype=float, delimiter=' ', skip_header=9, usecols=(0,1), unpack=True)
	
	fig, ax = plt.subplots()
	ax.semilogy(vU, vCounts, 'b.')
	# ax.plot(vU, line(fitparam,vU), 'r-')
	ax.set_title("Proportional counter with " + source + " sample")
	ax.set_xlabel("Voltage (V)")
	ax.set_ylabel("Counts")
	fig.savefig("Figures/" + "Proportional_characteristic_curve_"+source)
	fig.show()

