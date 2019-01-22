#-*- coding: utf-8 -*-
#
#@druckmessung.py:
#@author: Olexiy Fedorets
#@date: Thu 21.09.2017


import matplotlib.pyplot as plt
import numpy as np
import Praktikumsroutinen_DW as pr
import sys
sys.path.append("../../PraktikumPyLib/")
import PraktLib as pl
import random
from itertools import zip_longest
import time

####################################################################################
# DATA AND CONSTANTS:
####################################################################################

# berechnete Wellenlängen, in nm!
# Gruppe Philipp/Daniel(1):
lambdaPD = 538.38069996*10**-9; elambdaPD = 1.750144872*10**-9
# Gruppe Jonathan/Olex(2):
lambdaJO = 535.20062517761*10**-9; elambdaJO = 1.87309115585*10**-9

# Länge der Glasparzelle:
L = 0.01 # 10 mm
# DeltaN/DeltaP (Theorie):
dNdPtheo = 2.655*10**-7


# ausgemessenen Ordnungen:
m = np.linspace(1,8,8,endpoint=True,dtype=int)

# Daten, in hPa
# Gruppe Philipp/Daniel:
dataPD = np.array([ [866,775,681,603,477,388,292,191],
					[873,772,665,559,458,379,257,173],
					[882,790,695,614,506,402,293,196],
					[906,804,675,583,495,390,261,152],
					[874,775,694,609,490,391,298,174],
					[895,811,705,622,496,405,305,208],
					[868,755,662,569,468,360,264,150]	], dtype=float)
# Gruppe Jonathan/Olex:
dataJO = np.array([ [880,779,672,590,484,369,274,170],
					[862,763,663,561,475,353,258,165],
					[870,755,672,584,480,383,275,159],
					[889,766,676,565,484,398,274,173],
					[887,789,687,570,466,360,253,147],
					[889,797,679,581,477,389,289,176],
					[896,786,687,594,490,387,294,184],
					[887,782,690,591,494,376,284,163]	], dtype=float)

# pl.printAsLatexTable(dataJO, ["Ordnung %i" % col for col in range(dataJO.shape[1])])

####################################################################################


def shift(data,axis):
	diff = np.max(data[:,0]) - data[axis,0]
	data[axis,:] += diff
	return data

print("RAW DATA:\n")
print("dataPD:\n", dataPD)
print("dataJO:\n", dataJO)
print(dataJO.shape[0],dataPD.shape[0])
print(dataJO[0,1],dataJO.size,dataJO.shape)

# Rohdaten als scatterplot:
fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title("Rohdaten Gruppe 1", fontsize=20)	# PD
ax[1].set_title("Rohdaten Gruppe 2", fontsize=20)	# JO
ax[0].set_xlabel(r"Ordnung $m$", fontsize=20)
ax[1].set_xlabel(r"Ordnung $m$", fontsize=20)
ax[0].set_ylabel(r"Druck $[hPa]$", fontsize=20)
ax[1].set_ylabel(r"Druck $[hPa]$", fontsize=20)
ax[0].tick_params(axis='y', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[0].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[0].grid(True)
ax[1].grid(True)
plt.subplots_adjust(left=0.07,right=0.97,bottom=0.1,top=0.95,wspace=0.2)
# fig.tight_layout()

colors = pl.randomColors(max(dataPD.shape[0],dataJO.shape[0]))
# a,b = map(list,zip(*list(zip_longest(dataPD[:,0],dataJO[:,0],fillvalue=None))))
# print(list(zip(*list(zip_longest(dataPD[:,0],dataJO[:,0],fillvalue=None)))))
# for i,j in enumerate(zip_longest(dataPD[:,0],dataJO[:,0],fillvalue=None)):
for (i,j) in zip(range(dataPD.shape[0]), range(dataJO.shape[0])):
	ax[0].scatter(m,dataPD[i,:],c=colors[i],marker='*')
	ax[1].scatter(m,dataJO[j,:],c=colors[j],marker='*')


# Alle werte auf den größten ersten Messwert verschieben:
# print(shift(dataJO,0),shift(dataPD,0))
for (i,j) in zip(range(dataPD.shape[0]), range(dataJO.shape[0])):
	shift(dataPD,i)
	shift(dataJO,j)
print("shifted Data:\n",dataJO,dataPD)

# Mittelung pro Ordnung:
meanDataJO = np.mean(dataJO,axis=0,dtype=np.float64)
stdDataJO = np.std(dataJO,axis=0,dtype=np.float64)/np.sqrt(dataJO.shape[1])
# print("means,stds (JO):\n",meanDataJO,stdDataJO)
meanDataPD = np.mean(dataPD,axis=0,dtype=np.float64)
stdDataPD = np.std(dataPD,axis=0,dtype=np.float64)/np.sqrt(dataPD.shape[1])
# print("means,stds (JD):\n",meanDataPD,stdDataPD)

# Fehler auf ersten Messwert als arith. Mittel der restlichen Fehler annähern:
stdDataPD[0] = np.mean(stdDataPD[1:])
stdDataJO[0] = np.mean(stdDataJO[1:])
print("means,stds (JD):\n",meanDataPD,stdDataPD)
print("means,stds (JO):\n",meanDataJO,stdDataJO)
# yticks auf die Mittelwerte der Rohdaten setzen:
ax[0].set_yticks(meanDataPD)
ax[1].set_yticks(meanDataJO)
fig.savefig("Rohdaten.png",format="png",dpi=256)


# geshiftete Daten als scatterplot:
fig,ax = plt.subplots(1,2,figsize=(20,10))
fig.suptitle("Daten mit Fehler nach Verschiebung auf ersten Messwert")
ax[0].set_title("Gruppe 1", fontsize=20)	# PD
ax[1].set_title("Gruppe 2", fontsize=20)	# JO
ax[0].set_xlabel(r"Ordnung $m$", fontsize=20)
ax[1].set_xlabel(r"Ordnung $m$", fontsize=20)
ax[0].set_ylabel(r"Druck $[hPa]$", fontsize=20)
ax[1].set_ylabel(r"Druck $[hPa]$", fontsize=20)
ax[0].tick_params(axis='y', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[0].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[0].grid(True)
ax[1].grid(True)
for (i,j) in zip(range(dataPD.shape[0]), range(dataJO.shape[0])):
	# ax[0].scatter(m,dataPD[i,:],c=colors[i],marker='*')
	ax[0].errorbar(m,dataPD[i,:],yerr=stdDataPD,c=colors[i],marker='o',markersize=2,capsize=4,linestyle='None')
	# ax[1].scatter(m,dataJO[j,:],c=colors[j],marker='*')
	ax[1].errorbar(m,dataJO[i,:],yerr=stdDataJO,c=colors[i],marker='o',markersize=2,capsize=4,linestyle='None')

# fig.tight_layout()
plt.subplots_adjust(left=0.07,right=0.97,bottom=0.1,top=0.95,wspace=0.2)
# yticks auf die Mittelwerte der Rohdaten setzen:
ax[0].set_yticks(meanDataPD)
ax[1].set_yticks(meanDataJO)
fig.savefig("DatenShifted.png",format="png",dpi=256)

print(pl.separator(60))
print("Gruppe 1/PD:")
t0 = time.time()
a,ea,b,eb,chiq_ndof = pr.residuen(m,meanDataPD,0,stdDataPD,r"$m$",r"$[hPa]$","Ordnung","Druck",
									r"Gruppe 1 - lineare Regression an $\Delta P = \frac{\lambda}{2L}\cdot\frac{\Delta P}{\Delta n}\cdot \Delta m$",
									k=6,l=800,o=4,p=-10,ftsize=20)
t1 = time.time()
print("residuen() exec time: ", t1-t0)

dNdP, estat, esys = lambdaPD/(a*2*L), lambdaPD*ea/(2*L*a**2), elambdaPD/(2*L*a)
print("DeltaN/DeltaP = %E, err = %E(stat) +/- %E(sys)" % (dNdP, estat, esys) )
print("Abweichung: %.3f" % (np.abs(np.abs(dNdP) - dNdPtheo)/np.sqrt(estat**2+esys**2)) )
print("Chi²/ndf = %.3f" % chiq_ndof)
print("Steigung: %.3f +/- %.3f" % (a,ea))
print("Achsenabschnitt: %.3f +/- %.3f" % (b,eb))

print(pl.separator(60))
print("Gruppe 2/JO:")
t0 = time.time()
a,ea,b,eb,chiq_ndof = pr.residuen(m,meanDataJO,0,stdDataJO,r"$m$",r"$[hPa]$","Ordnung","Druck",
									r"Gruppe 2 - lineare Regression an $\Delta P = \frac{\lambda}{2L}\cdot\frac{\Delta P}{\Delta n}\cdot \Delta m$",
									k=6,l=800,o=4,p=-7,ftsize=20)
t1 = time.time()
print("residuen() exec time: ", t1-t0)

dNdP, estat, esys = lambdaJO/(a*2*L), lambdaJO*ea/(2*L*a**2), elambdaJO/(2*L*a)
print("DeltaN/DeltaP = %E, err = %E(stat) +/- %E(sys)" % (dNdP, estat, esys) )
print("Abweichung: %.3f" % (np.abs(np.abs(dNdP) - dNdPtheo)/np.sqrt(estat**2+esys**2)) )
print("Chi²/ndf = %.3f" % chiq_ndof)
print("Steigung: %.3f +/- %.3f" % (a,ea))
print("Achsenabschnitt: %.3f +/- %.3f" % (b,eb))




plt.show()
