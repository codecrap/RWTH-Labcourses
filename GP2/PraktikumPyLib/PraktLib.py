#-*- coding: utf-8 -*-
#
#@PraktLib.py:	alle möglichen Funktionen und Beispiele, die bei Versuchsauswertungen hilfreich sein könnten
#@author: Olexiy Fedorets
#@date: Mon 18.09.2017

# @TODO: fitData(method="manual,leastsq,ODR,curvefit,..."), readLabFile ohne String,
#		 residuen, abweichung mit fehler, chiq berechnen

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp
import io
import random
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
	dnew = data.encode('utf-8')
	return np.genfromtxt(io.BytesIO(dnew))

def weightedMean(x, ex):
	# return np.average(x,weights=ex,returned=True) 	# this should be equivalent
	mean = np.sum(x/ex**2)/np.sum(1./ex**2)
	sigma = np.sqrt(1./np.sum(1./ex**2))
	return mean, sigma

def minToDeg(value):
	return value/60.

def degToSr (value):
	return value*(2.*np.pi/360.)

def srToDeg (value):
	return value*(360./(2.*np.pi))

def chiq(f,ydata,yerrors=1.,ndf=1.):
	chiq = np.sum(np.power((ydata-f)/yerrors,2))
	return chiq, chiq/ndf

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
	return 0

def randomColors(num):
	colors = []
	for i in range(num):
		colors.append("#%06X" % random.randint(0,0xFFFFFF))
	return colors

def separator(length):
	return "="*length





####################################################################################

def lineare_regression(x,y,ey):
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

	s   = sum(1./ey**2)
	sx  = sum(x/ey**2)
	sy  = sum(y/ey**2)
	sxx = sum(x**2/ey**2)
	sxy = sum(x*y/ey**2)
	delta = s*sxx-sx*sx
	b   = (sxx*sy-sx*sxy)/delta
	a   = (s*sxy-sx*sy)/delta
	eb  = sqrt(sxx/delta)
	ea  = sqrt(s/delta)
	cov = -sx/delta
	corr = cov/(ea*eb)
	chiq  = sum(((y-(a*x+b))/ey)**2)

	return(a,ea,b,eb,chiq,corr)


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
		sc = sum(y*cos(omega*t))/len(t)
		ss = sum(y*sin(omega*t))/len(t)
		amp[i] = sqrt(sc**2+ss**2)
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
	amp = abs(scipy.fftpack.fft(y))
	freq = scipy.fftpack.fftfreq(t.size,dt)
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
	val = 1./sqrt(2.)
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
