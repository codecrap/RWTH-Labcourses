#-*- coding: utf-8 -*-
#
#@activity.py: perform energy calibration of multi channel analyzer
#@author: Olexiy Fedorets
#@date: Wed 20.02.2019

import matplotlib as plt
import numpy as np
import datetime as dt
import sys
sys.path.append("./../../")
import PraktLib as pl



datapath = "./Data/"
vProbes = ["Co","Cs","Eu","Na"]
FILE_POSTFIX = "_calibration.TKA"

mFile = []
for i, probe in enumerate(vProbes):
	mFile.append(np.genfromtxt(datapath+probe+FILE_POSTFIX,dtype=float,delimiter='\n',unpack=True))

vNoise =  np.genfromtxt(datapath+"Noise"+FILE_POSTFIX,dtype=float,delimiter='\n',unpack=True)


fig,ax = plt.subplots(2,2,figsize=(15,10))

for i,axis in enumerate(ax):
axis.plot(mFile[i],'b.')
ax[0].errorbar(x,y,xerr=xerr,yerr=yerr,fmt='.',color='b',capsize=capsize-1)
ax[0].set_title(title,fontsize=fontsize)
ax[0].set_xlabel(xlabel,fontsize=fontsize)
ax[0].set_ylabel(ylabel,fontsize=fontsize)
ax[0].legend(loc='lower right',fontsize=fontsize)
ax[0].grid(True)
ax[1].errorbar(x,residue,yerr=np.sqrt(yerr**2+fitparam[0]*xerr**2),fmt='x',color='b',capsize=capsize,
			label=r"$\frac{\chi^2}{ndf} = %.3f$" % np.around(chiq,3))
ax[1].axhline(0,color='r')
ax[1].set_title("Residuenverteilung",fontsize=fontsize)
ax[1].set_xlabel(xlabel,fontsize=fontsize)
ax[1].set_ylabel(res_ylabel,fontsize=fontsize)
ax[1].legend(loc='upper right',fontsize=fontsize)
ax[1].grid(True)
fig.tight_layout()
fig.savefig("Plots/"+title+".eps",format='eps',dpi=256)
