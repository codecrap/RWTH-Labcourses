#-*- coding: utf-8 -*-
#
#@xray_plots.py: plot raw data from xray source experiment
#@author: Olexiy Fedorets
#@date: Tue 09.03.2019


import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import uncertainties.unumpy as unp
from uncertainties import ufloat
import operator
from functools import reduce
import peakutils as pu


import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

DATAPATH = "Data/"
FILE_POSTFIX = ".mca"
FILE_PREFIX = "xray_"
vCALIBRATION_SAMPLE = ["ag","cu","fe","mo"]
vEMPTY_MEAS = ["leer_"+x for x in ["leer","folie","papier"] ]
vANALYSIS_SAMPLE = ["10ct","chip","chip2","battery","schnecke","stein","tab"]
vCOMPARISON_SAMPLE = ["pb"+x for x in ["","_20kV","_35kV","_PUR"] ]

# MCA CALIBRATION
fig, ax = plt.subplots(2,2)
ax = ax.ravel()

for i,sample in enumerate(vCALIBRATION_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "kal_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[1] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	# vData -= vNoise
	vCh = np.arange(0,len(vData))
	ax[i].semilogy(vCh, vData, 'b,')
	ax[i].set_xlabel('MCA Channel')
	ax[i].set_ylabel('Event counts')
	ax[i].set_title(sample)
	

fig.savefig("Figures/" + "XRay-calibration")



# SAMPLE ANALYSIS
fig, ax = plt.subplots(4, 2, figsize=(30,40))
ax = ax.ravel()

for i, sample in enumerate(vANALYSIS_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	# vData -= vNoise
	vCh = np.arange(0, len(vData))
	ax[i].semilogy(vCh, vData, 'b,')
	ax[i].set_xlabel('MCA Channel')
	ax[i].set_ylabel('Event counts')
	ax[i].set_title(sample)

fig.subplots_adjust(hspace=0.001,wspace=0.001)
fig.delaxes(ax[-1])
fig.savefig("Figures/" + "XRay-analysis")


# PUR and Voltage EFFECT COMPARISON
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()

for i, sample in enumerate(vCOMPARISON_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	# vData -= vNoise
	vCh = np.arange(0, len(vData))
	ax[i].semilogy(vCh, vData, 'b,')
	ax[i].set_xlabel('MCA Channel')
	ax[i].set_ylabel('Event counts')
	ax[i].set_title(sample.replace("_","\_"))										# escape the underscore to calm down the latex interpreter

fig.savefig("Figures/" + "XRay-comparison")

plt.show()
plt.close('all')