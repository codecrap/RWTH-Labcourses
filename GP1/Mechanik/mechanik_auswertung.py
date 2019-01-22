#-*- coding: utf-8 -*-
#
#@auswertung.py:
#@author: Olexiy Fedorets
#@date: Fri 27.10.2017


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sys
sys.path.append("../")
import PraktLib as pl

dpi = 512
figsize = (20,10)
peaksize = 12
fontsize = 20

def getPeriod(U,T):
	eps = 0.001
	t1 = T[np.min(np.where((U<eps) & (U>-eps)))]
	t2 = T[np.max(np.where((U<eps) & (U>-eps)))]
	print(t1,t2)
	period = (t2-t1)/(len(np.where((U<eps) & (U>-eps)))/2)
	return period

def findPeak(x,y,n=1):
	peakInds = []
	for i in range(n):
		maxYInd = np.where(y==max(y))[0]
		print(maxYInd)
		peakInds.append(maxYInd)
		y[maxYInd] = 0
	return peakInds

def plotPendel(t,U,axis,linestyle,title):
	axis.plot(t,U,linestyle)
	axis.set_xlabel("t [s]",fontsize=fontsize)
	axis.set_ylabel("U [V] (~Winkel)",fontsize=fontsize)
	axis.set_title(title,fontsize=fontsize)
	axis.grid(True)

def plotFFT(t,U,axis,linestyle):
	dt = (t[-1]-t[0])/(len(t)-1)
	freq, amp = pl.fourier_fft(t,U)
	amp = amp[freq>0]																# we dont need negative freq's and amp's, also the 0-th values are often rubbish
	freq = freq[freq>0]
	# peaks = scipy.signal.find_peaks_cwt(amp,np.arange(200,500))
	# peaks = pl.peak(freq,amp,0,20)
	# peaks = np.nanargmin(np.fabs(freq-peaks))
	# peaks = findPeak(freq, amp, n)
	# peaks = np.where(np.max(amp))
	peaks = pl.detect_peaks(amp,mph=10,mpd=1,threshold=100,edge='both')
	print(*freq[peaks])
	if peaks.size!=0:
		axis.plot(freq[0:2*np.max(peaks)],amp[0:2*np.max(peaks)],linestyle)
		axis.plot(freq[peaks],amp[peaks],'rx',ms=peaksize,
				label="Peaks (%s): " % linestyle[0] + ", ".join([r"$%.3f\pm\frac{%.3f}{\sqrt{12}} \, Hz$" % (np.around(freq[peak],3),dt) for peak in peaks]) )
	axis.set_xlabel("f [Hz]",fontsize=fontsize)
	axis.set_ylabel("Amplitude [rel.]",fontsize=fontsize)
	axis.set_title("Fourierspektrum",fontsize=fontsize)
	axis.legend(loc="upper right",fontsize=fontsize)
	axis.grid(True)



# Rauschmessung:
_, _, Urechts, Ulinks, _, _= pl.readLabFile("Data/rauschen.lab")
Rrauschen = np.nanmean(Urechts)
Lrauschen = np.nanmean(Ulinks)

# Periodendauer der Stange (links):
fig,ax = plt.subplots(2,1,figsize=figsize)
_, t, _, Ulinks, _, _= pl.readLabFile("Data/stange.lab")
Ulinks -= Lrauschen

plotPendel(t,Ulinks,ax[0],'g-',"Schwingung Pendelstange ohne Masse")
plotFFT(t,Ulinks,ax[1],'g.')

fig.tight_layout()
fig.savefig("Plots/Stange.eps",format='eps',dpi=dpi)


# Einfach-Pendel:
fig,ax = plt.subplots(2,1,figsize=figsize)
_, t, Urechts, _, _, _= pl.readLabFile("Data/pendel1.lab")
Urechts -= Rrauschen

plotPendel(t, Urechts, ax[0], 'g-', "Schwingung Einfach-Pendel")
plotFFT(t,Urechts,ax[1],'g.')

fig.tight_layout()
fig.savefig("Plots/Einfachpendel.eps",format='eps',dpi=dpi)


# Doppelpendel:
# Fall 1 (gleichsinnig):
fig,ax = plt.subplots(2,1,figsize=figsize)
_, t, Urechts, Ulinks, _, _= pl.readLabFile("Data/doppel_oben_fall1(2).lab")
Urechts -= Rrauschen
Ulinks -= Lrauschen

plotPendel(t, Urechts, ax[0], 'g-', "Doppelpendel Fall 1 - gleichsinnige Schwingung")
plotPendel(t, Ulinks, ax[0], 'b-', "Doppelpendel Fall 1 - gleichsinnige Schwingung")
plotFFT(t, Urechts, ax[1], 'g.')
plotFFT(t, Ulinks, ax[1], 'b.')

fig.tight_layout()
fig.savefig("Plots/Doppelpendel1.eps",format='eps',dpi=dpi)


# Fall 2 (gegensinnig):
fig,ax = plt.subplots(2,1,figsize=figsize)
_, t, Urechts, Ulinks, _, _= pl.readLabFile("Data/doppel_oben_fall2.lab")
Urechts -= Rrauschen
Ulinks -= Lrauschen

plotPendel(t, Urechts, ax[0], 'g-', "Doppelpendel Fall 2 - gegensinnige Schwingung")
plotPendel(t, Ulinks, ax[0], 'b-', "Doppelpendel Fall 2 - gegensinnige Schwingung")
plotFFT(t, Urechts, ax[1], 'g.')
plotFFT(t, Ulinks, ax[1], 'b.')

fig.tight_layout()
fig.savefig("Plots/Doppelpendel2.eps",format='eps',dpi=dpi)


# Fall 3 (Schwebung):
fig,ax = plt.subplots(2,1,figsize=figsize)
_, t, Urechts, Ulinks, _, _= pl.readLabFile("Data/doppel_oben_fall3.lab")
Urechts -= Rrauschen
Ulinks -= Lrauschen

plotPendel(t, Urechts, ax[0], 'g-', "Doppelpendel Fall 3 - Schwebung")
plotPendel(t, Ulinks, ax[0], 'b-', "Doppelpendel Fall 3 - Schwebung")
plotFFT(t, Urechts, ax[1], 'g.')
plotFFT(t, Ulinks, ax[1], 'b.')

fig.tight_layout()
fig.savefig("Plots/Doppelpendel3.eps",format='eps',dpi=dpi)




plt.show()
