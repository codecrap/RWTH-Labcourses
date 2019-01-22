#-*- coding: utf-8 -*-
#
#@PraktLib.py:	alle möglichen Funktionen und Beispiele, die bei Versuchsauswertungen hilfreich sein könnten
#@author: Olexiy Fedorets
#@date: Mon 18.09.2017

# @TODO: fitData(method="manual,leastsq,ODR,curvefit,..."), readLabFile ohne String,
#		 residuen, abweichung mit fehler, chiq berechnen

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.odr
import io
import random
import scipy.optimize as spopt
# import sys
# sys.path.append("../../PraktikumPyLib/")

def readLabFile(file):
	'''
	CASSY LAB Datei einlesen (Version fuer python3).

	Messdaten werden anhand von Tabulatoren identifiziert.

	Gibt ein numpy-Array zurueck.

	'''
	f = open(file)
	dataSectionStarted = False
	dataSectionEnded = False
	data = ''
	for line in f:
		if '\t' in line and not dataSectionEnded:
			data += line
			dataSectionStarted = True
		if not '\t' in line and dataSectionStarted:
			dataSectionEnded = True
	f.close()
	dnew = data.encode('utf-8',errors='ignore')
	return np.genfromtxt(io.BytesIO(dnew), unpack=True, dtype=float)

def weightedMean(x, ex):
	# return np.average(x,weights=ex,returned=True) 	# this should be equivalent
	mean = np.sum(x/ex**2)/np.sum(1./ex**2)
	sigma = np.sqrt(1./np.sum(1./ex**2))
	return mean, sigma

def minToDeg(value):
	return value/60.

def degToSr(value):
	return value*(2.*np.pi/360.)

def srToDeg(value):
	return value*(360./(2.*np.pi))

def chiq(f,ydata,yerrors=1.,ndf=1.):
	chiq = np.sum(np.power((ydata-f)/yerrors,2))
	return chiq, chiq/ndf

def deviation(expval,expval_err,trueval,trueval_err):
	return np.abs(trueval-expval)/np.sqrt(expval_err**2+trueval_err**2)

# @TODO
# This is made for Python3 ONLY !!
def printAsLatexTable(data,colTitles,rowTitles="",mathMode=True,decimals=2):
	# safety first :
	if len(colTitles) != data.shape[1]:
		print("Dimensions of data an colTitles don't match!")
		return -1
	if 0 != len(rowTitles) and len(rowTitles) != data.shape[0]-1:					# -1 because we don't put anything in upper left corner of the table
		print("Dimensions of data an rowTitles don't match!")
		return -2

	# we need the r-strings here to escape format placeholders like {} or \
	print("\n")
	print(r"\begin{table}[H]")
	print(r"\renewcommand{\arraystretch}{1.5}")
	print(r"\centering")
	print(r"	\begin{tabular}{|%s|}" % "|".join("c" for _ in colTitles))
	print(r"	\hline")
	print(r"	&" if 0!=len(rowTitles) else "	",
			" & ".join(str(colTitle) for colTitle in colTitles),
			r"\\")
	print(r"	\hline")

	# use of %g should be considered here, as we dont know how data will exactly look like
	if 0 != len(rowTitles):
		for row,rowTitle in enumerate(rowTitles):
			print("	%s" % str(rowTitle),
					" & ".join("${:.{prec}f}$".format(data[row,col], prec=decimals) if mathMode else "%s" % data[row,col] for col in range(data.shape[1]) ),
					r"\\")
			print("	\hline")
	else:
		for row in range(data.shape[0]):
			print("	",
					" & ".join("${:.{prec}f}$".format(data[row,col], prec=decimals) if mathMode else "%s" % data[row,col] for col in range(data.shape[1]) ),
					r"\\")
			print("	\hline")

	print(r"	\end{tabular}")
	print(r"\caption{ }")
	print(r"\label{table: }")
	print(r"\end{table}")
	print("\n")
	return 0

def randomColors(num):
	colors = []
	for i in range(num):
		colors.append("#%06X" % random.randint(0,0xFFFFFF))
	return colors

def separator(length):
	print("="*length)

def linreg_manual(x,y,ey):
	'''

	Lineare Regression.

	Parameters
	----------
	x : array_like
		x-Werte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte
	ey : array_like
		Fehler auf die y-Werte der Datenpunkte

	Diese Funktion benoetigt als Argumente drei Listen:
	x-Werte, y-Werte sowie eine mit den Fehlern der y-Werte.
	Sie fittet eine Gerade an die Werte und gibt die
	Steigung a und y-Achsenverschiebung b mit Fehlern
	sowie das chi^2 und die Korrelation von a und b
	als Liste aus in der Reihenfolge
	[a, ea, b, eb, chiq, cov].
	'''

	s   = np.sum(1./ey**2)
	sx  = np.sum(x/ey**2)
	sy  = np.sum(y/ey**2)
	sxx = np.sum(x**2/ey**2)
	sxy = np.sum(x*y/ey**2)
	delta = s*sxx-sx*sx
	b   = (sxx*sy-sx*sxy)/delta
	a   = (s*sxy-sx*sy)/delta
	eb  = np.sqrt(sxx/delta)
	ea  = np.sqrt(s/delta)
	cov = -sx/delta
	corr = cov/(ea*eb)
	chiq  = np.sum(((y-(a*x+b))/ey)**2)

	return(a,ea,b,eb,chiq,corr)

def plotFit(x,xerr,y,yerr,title="test",xlabel="",ylabel="",res_ylabel=r"$y - (a \cdot x + b)$",capsize=3,fontsize=20,show=True,method='leastsq'):
	# print(len(x),len(xerr),len(y),len(yerr),y.shape)
	line = lambda p,x: p[0]*x+p[1]


	if method=='leastsq':
		p0 = [0,0]	# start values
		chifunc = lambda p,x,xerr,y,yerr: (y-line(p,x))/np.sqrt(yerr**2+p[0]*xerr**2)	# p[0] = d/dx line()
		fitparam,cov,_,_,_ = spopt.leastsq(chifunc,p0,args=(x,xerr,y,yerr),full_output=True)
		# print(fitparam,cov)
		chiq = np.sum(chifunc(fitparam,x,xerr,y,yerr)**2) / (len(y)-len(fitparam))
		fitparam_err = np.sqrt(np.diag(cov)*chiq)									# leastsq returns the 'fractional covariance matrix'
		# print(chiq,fitparam_err)

	if method=='ODR':
		a_ini,ea_ini,b_ini,eb_ini,chiq_ini,corr_ini = linreg_manual(x,y,yerr)


		def f(B, x):
			return B[0]*x + B[1]

		model  = scipy.odr.Model(f)
		data   = scipy.odr.RealData(x, y, sx=xerr, sy=yerr)
		odr    = scipy.odr.ODR(data, model, beta0=[a_ini, b_ini])
		output = odr.run()
		ndof = len(x)-2
		chiq = output.res_var*ndof
		corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])

		fitparam = [output.beta[0],output.beta[1]]
		fitparam_err = [output.sd_beta[0],output.sd_beta[1]]

	if show:
		fig,ax = plt.subplots(2,1,figsize=(15,10))
		residue = y-line(fitparam,x)
		ax[0].plot(x,line(fitparam,x),'r-',
					label="Fit: $a = %.3f \pm %.3f$, \n     $b = %.3f \pm %.3f$"
							% (fitparam[0],fitparam_err[0],fitparam[1],fitparam_err[1]))
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
		fig.savefig("Plots/"+title+".pdf",format='pdf',dpi=256)

	return fitparam,fitparam_err,chiq

####################################################################################




def lineare_regression_xy(x,y,ex,ey):
	'''

	Lineare Regression mit Fehlern in x und y.

	Parameters
	----------
	x : array_like
		x-Werte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte
	ex : array_like
		Fehler auf die x-Werte der Datenpunkte
	ey : array_like
		Fehler auf die y-Werte der Datenpunkte

	Diese Funktion benoetigt als Argumente vier Listen:
	x-Werte, y-Werte sowie jeweils eine mit den Fehlern der x-
	und y-Werte.
	Sie fittet eine Gerade an die Werte und gibt die
	Steigung a und y-Achsenverschiebung b mit Fehlern
	sowie das chi^2 und die Korrelation von a und b
	als Liste aus in der Reihenfolge
	[a, ea, b, eb, chiq, cov].

	Die Funktion verwendet den ODR-Algorithmus von scipy.
	'''
	a_ini,ea_ini,b_ini,eb_ini,chiq_ini,corr_ini = lineare_regression(x,y,ey)

	def f(B, x):
		return B[0]*x + B[1]

	model  = scipy.odr.Model(f)
	data   = scipy.odr.RealData(x, y, sx=ex, sy=ey)
	odr    = scipy.odr.ODR(data, model, beta0=[a_ini, b_ini])
	output = odr.run()
	ndof = len(x)-2
	chiq = output.res_var*ndof
	corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])

	return output.beta[0],output.sd_beta[0],output.beta[1],output.sd_beta[1],chiq,corr


def fourier(t,y):
	'''

	Fourier-Transformation.

	Parameters
	----------
	t : array_like
		Zeitwerte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte

	Gibt das Fourierspektrum in Form zweier Listen (freq,amp)
	zurueck, die die Fourieramplituden als Funktion der zugehoerigen
	Frequenzen enthalten.
	'''
	dt = (t[-1]-t[0])/(len(t)-1)
	fmax = 0.5/dt
	step = fmax/len(t)
	freq=np.arange(0.,fmax,2.*step)
	amp = np.zeros(len(freq))
	i=0
	for f in freq:
		omega=2.*np.pi*f
		sc = sum(y*np.cos(omega*t))/len(t)
		ss = sum(y*np.sin(omega*t))/len(t)
		amp[i] = np.sqrt(sc**2+ss**2)
		i+=1
	return (freq,amp)


def fourier_fft(t,y):
	'''

	Schnelle Fourier-Transformation.

	Parameters
	----------
	t : array_like
		Zeitwerte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte

	Gibt das Fourierspektrum in Form zweier Listen (freq,amp)
	zurueck, die die Fourieramplituden als Funktion der zugehoerigen
	Frequenzen enthalten.
	'''
	dt = (t[-1]-t[0])/(len(t)-1)
	# print(dt,t.size)
	amp = abs(scipy.fftpack.fft(y))
	freq = scipy.fftpack.fftfreq(len(t),dt)
	# amp = np.fft.fft(y)
	# freq = np.fft.fftfreq(len(t))
	return (freq,amp)


def exp_einhuellende(t,y,ey,Sens=0.1):
	'''
	Exponentielle Einhuellende.

	Parameters
	----------
	t : array_like
		Zeitwerte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte
	ey : array_like
		Fehler auf die y-Werte der Datenpunkte
	Sens : float, optional
		Sensitivitaet, Wert zwischen 0 und 1

	Die Funktion gibt auf der Basis der drei Argumente (Listen
	mit t- bzw. dazugehoerigen y-Werten plus y-Fehler) der Kurve die
	Parameter A0 und delta samt Fehlern der Einhuellenden von der Form
	A=A0*exp(-delta*t) (Abfallende Exponentialfunktion) als Liste
	[A0, sigmaA0, delta, sigmaDelta] aus.
	Optional kann eine Sensibilitaet angegeben werden, die bestimmt,
	bis zu welchem Prozentsatz des hoechsten Peaks der Kurve
	noch Peaks fuer die Berechnung beruecksichtigt werden (default=10%).
	'''
	if not 0.<Sens<1.:
		raise ValueError('Sensibilitaet muss zwischen 0 und 1 liegen!')

	# Erstelle Liste mit ALLEN Peaks der Kurve
	Peaks=[]
	PeakZeiten=[]
	PeakFehler=[]
	GutePeaks=[]
	GutePeakZeiten=[]
	GutePeakFehler=[]
	if y[0]>y[1]:
		Peaks.append(y[0])
		PeakZeiten.append(t[0])
		PeakFehler.append(ey[0])
	for i in range(1,len(t)-1):
		if y[i] >= y[i+1] and \
		   y[i] >= y[i-1] and \
		   ( len(Peaks)==0 or y[i] != Peaks[-1] ): #handle case "plateau on top of peak"
		   Peaks.append(y[i])
		   PeakZeiten.append(t[i])
		   PeakFehler.append(ey[i])

	# Loesche alle Elemente die unter der Sensibilitaetsschwelle liegen
	Schwelle=max(Peaks)*Sens
	for i in range(0,len(Peaks)):
		if Peaks[i] > Schwelle:
			GutePeaks.append(Peaks[i])
			GutePeakZeiten.append(PeakZeiten[i])
			GutePeakFehler.append(PeakFehler[i])

	# Transformiere das Problem in ein lineares
	PeaksLogarithmiert = log(np.array(GutePeaks))
	FortgepflanzteFehler = np.array(GutePeakFehler) / np.array(GutePeaks)
	LR = lineare_regression(np.array(GutePeakZeiten),PeaksLogarithmiert,FortgepflanzteFehler)

	A0=exp(LR[2])
	sigmaA0=LR[3]*exp(LR[2])
	delta=-LR[0]
	sigmaDelta=LR[1]
	return(A0,sigmaA0,delta,sigmaDelta)

def untermenge_daten(x,y,x0,x1):
	'''
	Extrahiere kleinere Datensaetze aus (x,y), so dass x0 <= x <= x1
	'''
	xn=[]
	yn=[]
	for i,v in enumerate(x):
		if x0<=v<=x1:
			xn.append(x[i])
			yn.append(y[i])

	return (np.array(xn),np.array(yn))

def peak(x,y,x0,x1):
	'''
	Approximiere ein lokales Maximum in den Daten (x,y) zwischen x0 und x1.
	'''
	N = len(x)
	ymin = max(y)
	ymax = min(y)
	i1 = 0
	i2 = N-1
	for i in range(N):
		if x[i]>=x0:
			i1=i
			break
	for i in range(N):
		if x[i]>=x1:
			i2=i+1
			break
	for i in range(i1,i2):
		if y[i]>ymax:
			ymax=y[i]
		if y[i]<ymin:
			ymin=y[i]

	sum_y   = sum(y[i1:i2])
	sum_xy  = sum(x[i1:i2]*y[i1:i2])
	xm = sum_xy/sum_y
	return xm

def peakfinder_schwerpunkt(x,y):
	'''
	Finde Peak in den Daten (x,y).
	'''
	N = len(x)
	val = 1./np.sqrt(2.)
	i0=0
	i1=N-1
	ymax=max(y)
	for i in range(N):
		if y[i]>ymax*val:
			i0=i
			break
	for i in range(i0+1,N):
		if y[i]<ymax*val:
			i1=i
			break
	xpeak = peak(x,y,x[i0],x[i1])
	return xpeak


def gewichtetes_mittel(y,ey):
	'''
	Berechnet den gewichteten Mittelwert der gegebenen Daten.

	Parameters
	----------
	y : array_like
		Datenpunkte
	ey : array_like
		Zugehoerige Messunsicherheiten.

	Gibt den gewichteten Mittelwert samt Fehler zurueck.
	'''
	w = 1/ey**2
	s = sum(w*y)
	wsum = sum(w)
	xm = s/wsum
	sx = sqrt(1./wsum)

	return (xm,sx)



def residuen(x,y,ex,ey,ux,uy,Parameterx,Parametery,k=0,l=0,o=0,p=0,ftsize=15,ca=3,cr=3,mksizea=1,mksizer=1):
	'''
	Erstellt Residuenplot anhand von:
		x-Werte
			als np.array
		y-Werte
			als np.array
		Fehler(ex,ey) auf x und y
			  als np.array------Falls keine x-Fehler vorliegen einfach ex als 0 angeben
		ux,uy :
				Einheiten von x und y (str) als /$ux$ angeben
		ftsize:
				Schriftgöße
		ca:
			capsize Ausgleichsgerade
		cr:
			capsize Residuenplot
		mksize(a/r):
			Dicke der Punkte bzw Messwert-darstellung für a/(r) Ausgleichsgraph/(Residuenplot)
		Parameterx:
			die Abkürzung der x-Variable im Titel und auf der Achse
		Parametery:
			die Abkürzung der y-Variable im Titel und auf der Achse
		#ru:
			Stelle auf die die Werte im Plot gerundet werden sollen
		Diese Funktion vereint lineare_regression und lineare_regression_xy der praktikumsbibliothek und plottet.
		k:
			x-position der a-Werte im Plot
		l:
			y-Position der a-Werte im Plot
		o:
			x-Position der chiq-Werte
		p:
			y-Position der chiq-Werte
	'''

	if type(ex)==int and ex==0:
		 ex=0
		 #print(1)
		 #k=np.min(x)
		 #l=np.max(y)
		 a=lineare_regression(x,y,ey)[0]
		 ea=lineare_regression(x,y,ey)[1]
		 b=lineare_regression(x,y,ey)[2]
		 eb=lineare_regression(x,y,ey)[3]
		 chiq=lineare_regression(x,y,ey)[4]
		 chiq_ndof=chiq/(len(x)-2)
		 Ausgleichsgerade=a*x+b
		 #y-Achse vom Endplot
		 res=y-Ausgleichsgerade
		 #Fehlerresiduenplot
		 sres=np.sqrt(ey**2+(a*ex)**2)
		 """bestimmung y-Position des chiq/ndof im residuenplot"""
		 #hilfsvariablen für Text im Plot
		 h='a='+('{0:9.0f}').format(a)+'+/-'+('{0:9.0f}').format(ea)
		 i='b='+('{0:9.4f}').format(b)+'+/-'+('{0:9.4f}').format(eb)
		 j='chiq/ndof='+('{0:9.2f}').format(chiq_ndof)
		 fig1, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
		 ax0.errorbar(x, y,ey,0, 'o', ms = mksizea, capsize=ca)
		 ax0.plot(x, Ausgleichsgerade , "r")
		 ax0.set_title(Parametery +' gegen '+Parameterx +' (blau) und Ausgleichsgerade (rot)' , fontsize = ftsize)
		 ax0.grid()
		 ax0.annotate('{0} \n{1}'.format(h,i),xy=(k,l),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})
		 ax0.set_ylabel(Parametery+ uy, fontsize = ftsize)
		 ax1.set_xlabel(Parameterx+ ux, fontsize = ftsize)
		 ax1.set_ylabel(Parametery+" - (a*" + Parameterx +" + b)  " + uy, fontsize = ftsize)
		 ax1.errorbar(x, res, sres, 0, "o", ms = mksizer, capsize=cr)
		 ax1.plot(x, 0*x, "red")
		 ax1.annotate(j,xy=(o,p),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})
		 ax1.set_title("Residuenplot", fontsize = ftsize)
		 ax1.grid()
		 #print('Steigung',a,'+/-',ea)
		 #print('Achsenabschnitt',b,'+/-',eb)
		 #print('chiq',chiq_ndof)
		 return(a,ea,b,eb,chiq_ndof)
	elif type(ex)==np.ndarray:
		#print(2)
		ex=ex
		"""bestimmung der Koordinaten für die Texte im Plot"""
		a=lineare_regression_xy(x,y,ex,ey)[0]
		ea=lineare_regression_xy(x,y,ex,ey)[1]
		b=lineare_regression_xy(x,y,ex,ey)[2]
		eb=lineare_regression_xy(x,y,ex,ey)[3]
		chiq=lineare_regression_xy(x,y,ex,ey)[4]
		chiq_ndof=chiq/(len(x)-2)
		Ausgleichsgerade=a*x+b
		#y-Achse vom Endplot
		res=y-Ausgleichsgerade
		#hilfsvariablen für Text im Plot
		h='a='+('{0:9.4f}').format(a)+'+/-'+('{0:9.4f}').format(ea)
		i='b='+('{0:9.4f}').format(b)+'+/-'+('{0:9.4f}').format(eb)
		j='chiq/ndof='+('{0:9.4f}').format(chiq_ndof)
		#Fehlerresiduenplot
		sres=np.sqrt(ey**2+(a*ex)**2)
		"""bestimmung y-Position des chiq/ndof im residuenplot"""
		fig1, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
		ax0.errorbar(x, y, ey, ex, 'o', ms = mksizea, capsize=ca)
		ax0.plot(x, Ausgleichsgerade , "r")
		ax0.set_title(Parametery +' gegen '+Parameterx +' (blau) und Ausgleichsgerade (rot)' , fontsize = ftsize)
		ax0.grid()
		ax0.annotate('{0} \n{1}'.format(h,i),xy=(k,l),fontsize=15,bbox={'facecolor':'white','alpha':0.5,'pad':4})
		ax0.set_ylabel(Parametery+ uy, fontsize = ftsize)
		ax1.set_xlabel(Parameterx+ ux, fontsize = ftsize)
		ax1.set_ylabel(Parametery+" - (a*" + Parameterx +" + b)"+  uy, fontsize = ftsize)
		ax1.errorbar(x, res, sres, 0, "o", ms = mksizer, capsize=cr)
		ax1.plot(x, 0*x,"red")
		ax1.annotate(j,xy=(o,p),fontsize=15,bbox={'facecolor':'white','alpha':0.5,'pad':4})
		ax1.set_title("Residuenplot", fontsize = ftsize)
		ax1.grid()
		#verhindern des überschneidens durch erhöhen von hspace bei ausführung mit python console nicht notwendig
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
		#print("Steigung",a,"+/-",ea)
		#print('Achsenabschnitt',b,'+/-',eb)
		#print('chiq/ndof',chiq_ndof)
		return(a,ea,b,eb,chiq_ndof)

	else:
		print('incorrect data type error x')

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
				 kpsh=False, valley=False, show=False, ax=None):

	"""Detect peaks in data based on their amplitude and other features.

	Parameters
	----------
	x : 1D array_like
		data.
	mph : {None, number}, optional (default = None)
		detect peaks that are greater than minimum peak height.
	mpd : positive integer, optional (default = 1)
		detect peaks that are at least separated by minimum peak distance (in
		number of data).
	threshold : positive number, optional (default = 0)
		detect peaks (valleys) that are greater (smaller) than `threshold`
		in relation to their immediate neighbors.
	edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
		for a flat peak, keep only the rising edge ('rising'), only the
		falling edge ('falling'), both edges ('both'), or don't detect a
		flat peak (None).
	kpsh : bool, optional (default = False)
		keep peaks with same height even if they are closer than `mpd`.
	valley : bool, optional (default = False)
		if True (1), detect valleys (local minima) instead of peaks.
	show : bool, optional (default = False)
		if True (1), plot data in matplotlib figure.
	ax : a matplotlib.axes.Axes instance, optional (default = None).

	Returns
	-------
	ind : 1D array_likeiui
		indeces of the peaks in `x`.

	Notes
	-----
	The detection of valleys instead of peaks is performed internally by simply
	negating the data: `ind_valleys = detect_peaks(-x)`

	The function can handle NaN's

	See this IPython Notebook [1]_.

	References
	----------
	.. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

	Examples
	--------
	>>> from detect_peaks import detect_peaks
	>>> x = np.random.randn(100)
	>>> x[60:81] = np.nan
	>>> # detect all peaks and plot data
	>>> ind = detect_peaks(x, show=True)
	>>> print(ind)

	>>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	>>> # set minimum peak height = 0 and minimum peak distance = 20
	>>> detect_peaks(x, mph=0, mpd=20, show=True)

	>>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
	>>> # set minimum peak distance = 2
	>>> detect_peaks(x, mpd=2, show=True)

	>>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	>>> # detection of valleys instead of peaks
	>>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

	>>> x = [0, 1, 1, 0, 1, 1, 0]
	>>> # detect both edges
	>>> detect_peaks(x, edge='both', show=True)

	>>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
	>>> # set threshold = 2
	>>> detect_peaks(x, threshold = 2, show=True)
	"""

	x = np.atleast_1d(x).astype('float64')
	if x.size < 3:
		return np.array([], dtype=int)
	if valley:
		x = -x
	# find indices of all peaks
	dx = x[1:] - x[:-1]
	# handle NaN's
	indnan = np.where(np.isnan(x))[0]
	if indnan.size:
		x[indnan] = np.inf
		dx[np.where(np.isnan(dx))[0]] = np.inf
	ine, ire, ife = np.array([[], [], []], dtype=int)
	if not edge:
		ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
	else:
		if edge.lower() in ['rising', 'both']:
			ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
		if edge.lower() in ['falling', 'both']:
			ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
	ind = np.unique(np.hstack((ine, ire, ife)))
	# handle NaN's
	if ind.size and indnan.size:
		# NaN's and values close to NaN's cannot be peaks
		ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
	# first and last values of x cannot be peaks
	if ind.size and ind[0] == 0:
		ind = ind[1:]
	if ind.size and ind[-1] == x.size-1:
		ind = ind[:-1]
	# remove peaks < minimum peak height
	if ind.size and mph is not None:
		ind = ind[x[ind] >= mph]
	# remove peaks - neighbors < threshold
	if ind.size and threshold > 0:
		dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
		ind = np.delete(ind, np.where(dx < threshold)[0])
	# detect small peaks closer than minimum peak distance
	if ind.size and mpd > 1:
		ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
		idel = np.zeros(ind.size, dtype=bool)
		for i in range(ind.size):
			if not idel[i]:
				# keep peaks with the same height if kpsh is True
				idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
					& (x[ind[i]] > x[ind] if kpsh else True)
				idel[i] = 0  # Keep current peak
		# remove the small peaks and sort back the indices by their occurrence
		ind = np.sort(ind[~idel])

	if show:
		if indnan.size:
			x[indnan] = np.nan
		if valley:
			x = -x
		_plot(x, mph, mpd, threshold, edge, valley, ax, ind)

	return ind






   #
   #
   # import numpy as np
   # from numpy import sqrt,sin,cos,log,exp
   # import scipy
   # import scipy.fftpack
   # import scipy.odr
   #
   # import numpy as np
   # import matplotlib.pyplot as plt
   # from pylab import *
   #
   # def sigmagew(liste):
   #  k=0
   #  for i in range(len(liste)):
   #   k+=1/liste[i]**2
   #  sigma= sqrt(1/k)
   #  return sigma
   #
   # def meangew(y,sigma):
   #  k=0
   #  for i in range(len(sigma)):
   #   k+=y[i]/sigma[i]**2
   #  mean=k*(sigmagew(sigma)**2)
   #  return mean
   #
   # def fourier_fft(t,y):
   #  '''
   #
   #  Schnelle Fourier-Transformation.
   #
   #  Parameters
   #  ----------
   #  t : array_like
   #   Zeitwerte der Datenpunkte
   #  y : array_like
   #   y-Werte der Datenpunkte
   #
   #  Gibt das Fourierspektrum in Form zweier Listen (freq,amp)
   #  zurueck, die die Fourieramplituden als Funktion der zugehoerigen
   #  Frequenzen enthalten.
   #  '''
   #  dt = (t[-1]-t[0])/(len(t)-1)
   #  amp = abs(scipy.fftpack.fft(y))
   #  freq = scipy.fftpack.fftfreq(len(t),dt)
   #  return (freq,amp)
   #
   # def fourier(t,y):
   #  '''
   #
   #  Fourier-Transformation.
   #
   #  Parameters
   #  ----------
   #  t : array_like
   #   Zeitwerte der Datenpunkte
   #  y : array_like
   #   y-Werte der Datenpunkte
   #
   #  Gibt das Fourierspektrum in Form zweier Listen (freq,amp)
   #  zurueck, die die Fourieramplituden als Funktion der zugehoerigen
   #  Frequenzen enthalten.
   #  '''
   #
   #  dt = (t[-1]-t[0])/(len(t)-1)
   #  fmax = 0.5/dt
   #  step = fmax/len(t)
   #  freq=np.arange(0.,fmax,2.*step)
   #  amp = np.zeros(len(freq))
   #  i=0
   #  for f in freq:
   #   omega=2.*np.pi*f
   #   sc =0
   #   ss =0
   #
   #   for j in range(len(t)):
   #    sc += (y[j]*cos(omega*t[j]))/len(t)
   #    ss += (y[j]*sin(omega*t[j]))/len(t)
   #
   #
   #
   #   amp[i] = sqrt(sc**2+ss**2)
   #   i+=1
   #  return (freq,amp)
   #
   #
   # def peaks(y, Sens):
   #  Peaks=[]
   #  GutePeaks=[]
   #  Indexes=[]
   #  GuteIndexes=[]
   #
   #
   #  for i in range(2,len(y)-3):
   #   if abs(y[i]) >= abs(y[i+1]) and \
   #   abs(y[i]) >= abs(y[i-1]) and \
   #   abs(y[i]) >= abs(y[i+2]) and \
   #   abs(y[i]) >= abs(y[i-2]) and \
   #   abs(y[i]) >= abs(y[i+3]) and \
   #   abs(y[i]) >= abs(y[i-3]) and \
   #   ( len(Peaks)==0 or y[i] != Peaks[-1] ): #handle case "plateau on top of peak"
   #   Peaks.append(y[i])
   #   Indexes.append(i)
   #
   #
   #  # Loesche alle Elemente die unter der Sensibilitaetsschwelle liegen
   #  Schwelle=max(Peaks)*Sens
   #  for i in range(0,len(Peaks)):
   #   if abs(Peaks[i]) > abs(Schwelle):
   #    GutePeaks.append(Peaks[i])
   #    GuteIndexes.append(i)
   #
   #  ind=[]
   #  for i in range(len(GuteIndexes)):
   #   ind.append(Indexes[GuteIndexes[i]])
   #
   #  return ind
   #
   # def differenz(x):
   #  diff = []
   #  for i in range(len(x)-1):
   #   d = abs(x[i+1]-x[i])
   #   diff.append(d)
   #  return diff
   #
   # def skiplines (text, n):    # n (Anzahl an Zeilen zu skippen)
   #  for i in range(n):
   #   text.readline()
   #
   # def spalten (text, n):   # n (Welche Spalte soll generiert werden)
   #  spalte = []
   #  for i in range(800):
   # bumper = text.readline()
   # zeile = bumper.split('\t') # gespaltene Zeile
   # spalte.append(zeile [n-1])
   #  return spalte
   #










	# annotate:
	# ax[0][0].annotate("Guete: Q = U_0/U_cross = %.2f" % (Ucross/U0), xy=(0.65,0.5), xycoords='axes fraction')
	# plt.xticks(list(plt.xticks()[0]) + extraticks)
	# lines = plt.plot(x,y)
	# ax = lines[0].axes
	# ax.set_xticks(list(ax.get_xticks()) + extraticks)

	# plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.1)         # just to make better use of space
	# plt.get_current_fig_manager().window.showMaximized()


	# QsG1,sigmaQsG1 = map(list,zip(*resultsG1))											# so macht man aus einer liste von tuplen zwei listen von jeweils allen tuple[0] und tuple[1]
   # QsG2,sigmaQsG2 = map(list,zip(*resultsG2))
# >>> pairs.sort(key=lambda pair: pair[1])
# >>> pairs.sort(key=lambda pair: pair[1])
