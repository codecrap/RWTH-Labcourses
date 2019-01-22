# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:09:25 2017

@author: tobia
"""

import numpy as np
from numpy import sqrt,sin,cos,log,exp
import scipy
import scipy.fftpack
import scipy.odr
import numpy as np
import matplotlib.pyplot as plt
from pylab import *












def sigmagew(liste):
    k=0
    for i in range(len(liste)):
        k+=1/liste[i]**2
    sigma= sqrt(1/k)
    return sigma
def meangew(y,sigma):
    k=0
    for i in range(len(sigma)):
        k+=y[i]/sigma[i]**2
    mean=k*(sigmagew(sigma)**2)
    return mean
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
    freq = scipy.fftpack.fftfreq(len(t),dt)
    return (freq,amp)
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
        sc =0
        ss =0
        for j in range(len(t)):
            sc += (y[j]*cos(omega*t[j]))/len(t)
            ss += (y[j]*sin(omega*t[j]))/len(t)
       
        
        
        amp[i] = sqrt(sc**2+ss**2)
        i+=1
    return (freq,amp)
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
    s = 0
    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    
    for i in range(len(ey)):
        s   += 1./ey[i]**2
        sx  += x[i]/ey[i]**2
        sy  += y[i]/ey[i]**2
        sxx += x[i]**2/ey[i]**2
        sxy += x[i]*y[i]/ey[i]**2
        
    delta = s*sxx-sx*sx
    b   = (sxx*sy-sx*sxy)/delta
    a   = (s*sxy-sx*sy)/delta
    eb  = sqrt(sxx/delta)
    ea  = sqrt(s/delta)
    cov = -sx/delta
    corr = cov/(ea*eb)
    
    chiq = 0
    
    for i in range(len(ey)):
        chiq  += (((y[i]-(a*x[i]+b))/ey[i])**2)
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
def error(y):
    ey = []
    for i in range (len(y)):
        ey.append(0.01*y[i]+0.05)
    return ey
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
def peaks(y, Sens):
    Peaks=[]
    GutePeaks=[]
    Indexes=[]
    GuteIndexes=[]
    
    
    for i in range(2,len(y)-3):
        if abs(y[i]) >= abs(y[i+1]) and \
           abs(y[i]) >= abs(y[i-1]) and \
           abs(y[i]) >= abs(y[i+2]) and \
           abs(y[i]) >= abs(y[i-2]) and \
           abs(y[i]) >= abs(y[i+3]) and \
           abs(y[i]) >= abs(y[i-3]) and \
           abs(y[i]) >= abs(y[i+4]) and \
           abs(y[i]) >= abs(y[i-4]) and \
           abs(y[i]) >= abs(y[i+5]) and \
           abs(y[i]) >= abs(y[i-5]) and \
           abs(y[i]) >= abs(y[i+6]) and \
           abs(y[i]) >= abs(y[i-6]) and \
           ( len(Peaks)==0 or y[i] != Peaks[-1] ): #handle case "plateau on top of peak"
           Peaks.append(y[i])
           Indexes.append(i)
           
    # Loesche alle Elemente die unter der Sensibilitaetsschwelle liegen
    Schwelle=max(Peaks)*Sens
    for i in range(0,len(Peaks)):
        if abs(Peaks[i]) > abs(Schwelle):
            GutePeaks.append(Peaks[i])
            GuteIndexes.append(i)
    
    ind=[]
    for i in range(len(GuteIndexes)):
        ind.append(Indexes[GuteIndexes[i]])
    
    return ind
def differenz(x):
    diff = []
    for i in range(len(x)-1):
        d = abs(x[i+1]-x[i])
        diff.append(d)
    return diff
def skiplines (text, n):    # n (Anzahl an Zeilen zu skippen)
    for i in range(n):
        text.readline()
        
def spalten (text, n, x):   # n (Welche Spalte soll generiert werden)
    spalte = []
    for i in range(x):
         bumper = text.readline()
         zeile = bumper.split('\t') # gespaltene Zeile
         spalte.append(zeile [n-1])
    return spalte


## Abgleich der Frequenzen
f = open('Fourier trafo.lab','r')
skiplines(f, 52)
time = []
buffer = spalten (f,2, 800)
l = len(buffer)
for i in range(l):
    time.append(float (buffer[i]))
f.close()

g = open('Fourier trafo.lab','r')
skiplines(g, 52)
voltage1 = []
buffer = spalten (g,3,800)
l = len(buffer)
for i in range(l):
    voltage1.append(float (buffer[i]))
g.close()

h = open('Fourier trafo.lab','r')
skiplines(h, 52)
voltage2 = []
buffer = spalten (h,4,800)
l = len(buffer)
for i in range(l):
    voltage2.append(float (buffer[i]))
h.close()

offset1 = np.mean(voltage1)
offset2 = np.mean(voltage2)
for i in range(len(voltage1)):
    voltage1[i]=voltage1[i]-offset1
    voltage2[i]=voltage2[i]-offset2


vpeaks = []
vpeaks = peaks(voltage1,0.1)
print(vpeaks)
for i in range(len(vpeaks)):
    plt.plot(time[vpeaks[i]], voltage1[vpeaks[i]], 'g+')
    
wpeaks = []
wpeaks = peaks(voltage2,0.1)
print(wpeaks)
for i in range(len(wpeaks)):
    plt.plot(time[wpeaks[i]], voltage2[wpeaks[i]], 'g+')
    
plt.plot(time, voltage1,'r-')
plt.plot(time, voltage2,'b-')
plt.grid()
plt.axis([0,8,-0.4, 0.4])
plt.show()
plt.savefig("Frequenzen.jpg", format='jpg', dpi=256)
plt.close()

## Bestimmung der Periodendauer

a = open('MessreihePeriode2.lab','r')
skiplines(a, 52)
time = []
buffer = spalten (a,2,10000)
l = len(buffer)
for i in range(l):
    time.append(float (buffer[i]))
a.close()

b = open('MessreihePeriode2.lab','r')
skiplines(b, 52)
voltage3 = []
buffer = spalten (b,3,10000)
l = len(buffer)
for i in range(l):
    voltage3.append(float (buffer[i]))
b.close()

offset3 = np.mean(voltage3)
for i in range(len(voltage3)):
    voltage3[i]=voltage3[i]-offset3

peaks3 = []
peaks3 = peaks(voltage3,0.1)
print(peaks3)
for i in range(len(peaks3)):
    plt.plot(time[peaks3[i]], voltage3[peaks3[i]], 'g+')
amp = scipy.fftpack.fft(voltage3)
freq = scipy.fftpack.fftfreq(len(time),(time[-1]-time[0])/(len(time)-1))
plt.close()

plt.plot(freq[0:1000],amp[0:1000],'cx')
plt.show()

plt.close()


plt.plot(time, voltage3,'r-')
plt.grid()
plt.axis([0,25,-0.25, 0.25])
plt.show()
plt.close()