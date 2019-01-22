#-*- coding: utf-8 -*-
#
#@auswertung_gitarre.py:
#@author: Olexiy Fedorets
#@date: Sat 18.11.2017


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import PraktLib as pl

fontsize = 20

# Peaks beim Anschlag der Bünde 0-9, in Hz
seiteA = np.array([110.4, 116.6, 123.7, 130.9, 138.6, 146.7, 155.6, 164.5, 174.1, 183.9])
seiteE = np.array([164.3, 174.4, 184.1, 194.2, 205.3, 220.3, 233.6, 246.6, 260.7, 276.6])
peakSigma = 1/0.8	# Abstand zwischen 2 Frequenzen in der FFT bei Messzeit 800ms

# Längen der Seite in den Bünden 0-9, in m
L = np.array([0.652,0.616,0.582,0.549,0.519,0.49,0.462,0.436,0.412,0.389])
LSigma = 0.001	# Maßband Unsicherheit 1mm

fitparam,fitparam_err,chiq = pl.plotFit(1/(2*L),LSigma,seiteA,peakSigma,
			title="Frequenzen beim Anschlag der A-Seite",xlabel=r"1/2L [1/m]",ylabel=r"f [Hz]")
print(fitparam,fitparam_err,chiq)

fitparam,fitparam_err,chiq = pl.plotFit(1/(2*L),LSigma,seiteE,peakSigma,
			title="Frequenzen beim Anschlag der E-Seite",xlabel=r"1/2L [1/m]",ylabel=r"f [Hz]")
print(fitparam,fitparam_err,chiq)

# Plot Frequenzspektren
files = np.array(["A-0,1.lab","A-0,2.lab","A-0,5.lab","A-0,25.lab","A-0,33.lab"])
for file in files:
	_,t,U = pl.readLabFile("./Data/gitarre/"+file)
	freq, amp = pl.fourier_fft(t,U)
	chooseWindow = np.logical_and(freq>0,freq<4000)
	amp = amp[chooseWindow]
	freq = freq[chooseWindow]
	fig,ax = plt.subplots(figsize=(15,10))
	ax.semilogy(freq,amp,'b-',label="A-Seite bei %s L" % file[2:-4])
	ax.set_xticks(np.arange(110,4000,220))
	ax.set_xlabel("f [Hz]",fontsize=fontsize)
	ax.set_ylabel("Amplitude [rel.]",fontsize=fontsize)
	ax.set_title("Frequenzspektrum der A-Seite abgegriffen bei %s L" % file[2:-4],fontsize=fontsize)
	ax.legend(loc="upper right",fontsize=fontsize)
	ax.grid(True)
	fig.tight_layout()
	fig.savefig("./Plots/"+file[:-4]+".pdf",format='pdf',dpi=256)





plt.show()
# def meanDif(data):
# 	diffs = []
# 	for i in range(0,len(data)-1):
# 		diffs.append(data[i+1]-data[i])
# 	return np.mean(diffs)
#
# print(meanDif(seiteA),meanDif(seiteE), np.mean(np.diff(seiteA)))


# a,ea,b,eb,chiq = pl.residuen(1/(2*L),seiteA,np.ones(len(L))*LSigma,np.ones(len(seiteA))*peakSigma,"d", "d", "Parameterx", "Parametery")
# print(a,ea,b,eb,chiq)
# a,ea,b,eb,chiq = pl.residuen(1/(2*L),seiteE,np.ones(len(L))*LSigma,np.ones(len(seiteE))*peakSigma,"d", "d", "Parameterx", "Parametery")
# print(a,ea,b,eb,chiq)
