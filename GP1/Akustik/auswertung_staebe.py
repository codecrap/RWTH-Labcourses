#-*- coding: utf-8 -*-
#
#@auswertung.py:
#@author: Olexiy Fedorets
#@date: Fri 17.11.2017


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import PraktLib as pl

fontsize = 20

# Frequenzen der Peaks bei der Schwingung in verschiedenen Metallstäben, in Hz
metals = ["Br","FE","CU","AL"]
Br_peaks = [1293.79, 1293.52, 1293.68, 1293.29, 1293.24, 1293.54, 1293.69, 1293.87, 1293.67, 1293.96]
FE_peaks = [1883.43, 1883.97, 1875.33, 1885.42, 1884.15, 1870.04, 1897.23, 1884.77, 1885.01, 1884.40]	# höchste schwankungen, peaks teilweise kleiner als störpeaks!
CU_peaks = [1511.39, 1513.91, 1512.01, 1513.00, 1511.51, 1513.89, 1513.44, 1513.41, 1512.96, 1513.11]
AL_peaks = [1922.86, 1923.41, 1923.53, 1923.13, 1923.22, 1923.57, 1923.74, 1922.41, 1923.48, 1923.45]
peakdata = [Br_peaks, FE_peaks, CU_peaks, AL_peaks]

def mean(data):
	mean = np.mean(data)
	msigma = np.std(data)/np.sqrt(len(data))
	return mean,msigma

peaks = []
for metal in peakdata:
	peaks.append(mean(metal))
print(peaks)

for i,metal in enumerate(metals):
	print(metal, "peak: %.2f \pm %.2f Hz" % (peaks[i][0],peaks[i][1]))

# Plot der Peakverteilung
fig,ax = plt.subplots(figsize=(15,10))
colors = pl.randomColors(len(peakdata))
for i,peaks in enumerate(peakdata):
	ax.plot(np.arange(len(peaks)),peaks,'.',color=colors[i],ms=15,label="Peaks von "+metals[i])
ax.set_xlabel("n",fontsize=fontsize)
ax.set_ylabel("Frequenz [Hz]",fontsize=fontsize)
ax.set_title("Verteilung der Peaks bei 10 Anschlägen",fontsize=fontsize)
ax.legend(loc="right",fontsize=fontsize)
ax.grid(True)
fig.tight_layout()
fig.savefig("Plots/Peakverteilung.pdf",format='pdf',dpi=256)

# Durchmesser in mm:
Br_D = 11.5 + np.array([0.48,0.47,0.485,0.47,0.47,0.47])
FE_D = 11.5 + np.array([0.46,0.47,0.465,0.47,0.47,0.465])
CU_D = 11.5 + np.array([0.46,0.46,0.46,0.46,0.46,0.465])
AL_D = 12.0 + np.array([0.05,0.07,0.06,0.065,0.06,0.06])
Ds = [Br_D,FE_D,CU_D,AL_D]

Dmeans = []
for d in Ds:
	Dmeans.append(mean(d))
print(Dmeans)

for i,metal in enumerate(metals):
	print(metal, "D: %.3f \pm %.3f mm" % (Dmeans[i][0],Dmeans[i][1]))

# # Plots von ausgewählten Schwingungen:
# def plotMetal(file,title,col):
# 	n, t, U = pl.readLabFile(file)
# 	fig,ax = plt.subplots(2,1)
# 	ax[0].plot(t,U,col,label="Peaks von "+metals[i])
# 	ax[0].set_xlabel("Zeit [s]",fontsize=fontsize)
# 	ax[0].set_ylabel("U [V]",fontsize=fontsize)
# 	ax[0].set_title(title,fontsize=fontsize)
# 	ax[0].legend(loc="upper right",fontsize=fontsize)
# 	ax[0].grid(True)
# 	fig.savefig(.eps",format='eps',dpi=256)


plt.show()
