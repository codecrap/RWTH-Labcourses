# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:27:30 2017

@author: tobia
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:34:42 2017

@author: tobia
"""
from numpy import sqrt,sin,cos,log,exp
import scipy
import scipy.fftpack
import scipy.odr
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import io
#import PraktLib as pl

# Definition der Funktionen
def skiplines (text, n):    # n (Anzahl an Zeilen zu skippen)
    for i in range(n):
        text.readline()

def abs(x):    #Betrag nehmen
         if x >= 0:
             return x
         n = math.sqrt(x ** 2)
         return abs(type(x)(n))

def spalten (text, n,x):   # n (Welche Spalte soll generiert werden)
    spalte = []
    for i in range(x):
         bumper = text.readline()
         zeile = bumper.split('\t') # gespaltene Zeile
         spalte.append(zeile [n])
    return spalte

def peak(x,y,x0,x1):
    '''
    Approximiere ein lokales Maximum in den Daten (x,y) zwischen x0 und x1.
    '''
    N = len(x)
    i1 = 0
    i2 = N-1
    for i in range(N):
       if x[i]>=x0:
         i1=i
         break
    for i in range(N):
       if x[i]>x1:
         i2=i
         break

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

    #s   = sum(1./ey**2)
    s=0
    for i in range(len(ey)):
        s+=1/(ey[i]**2)
    #sx  = sum(x/ey**2)
    sx=0
    for i in range(len(ey)):
        sx+=x[i]/(ey[i]**2)
    #sy  = sum(y/ey**2)
    sy=0
    for i in range(len(ey)):
        sy+=y[i]/(ey[i]**2)
    #sxx = sum(x**2/ey**2)
    sxx=0
    for i in range(len(ey)):
        sxx+=(x[i])**2/(ey[i]**2)
   # sxy = sum(x*y/ey**2)
    sxy=0
    for i in range(len(ey)):
        sxy+=(x[i]*y[i])/(ey[i]**2)

    delta = s*sxx-sx*sx
    b   = (sxx*sy-sx*sxy)/delta
    a   = (s*sxy-sx*sy)/delta
    eb  = sqrt(sxx/delta)
    ea  = sqrt(s/delta)
    cov = -sx/delta
    corr = cov/(ea*eb)
    #chiq  = sum(((y-(a*x+b))/ey)**2)
    chiq=0
    for i in range(len(ey)):
        chiq+=((y[i]-(a*x[i]+b))/ey[i])**2
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
        sc =0
        ss =0
        for j in range(len(t)):
            sc += (y[j]*cos(omega*t[j]))/len(t)
            ss += (y[j]*sin(omega*t[j]))/len(t)



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
    amp = np.abs(scipy.fftpack.fft(y))
    freq = scipy.fftpack.fftfreq(len(t),dt)

    print(freq,amp)
    return (freq,amp)
#Einlesen der Messwerte
#%%

An=0
E=13000 #vorher 14019 aber wegen verfälschung des w verkürzt

#def abs(x):    #Betrag nehmen
 #        if x >= 0:
  #           return x
   #      n = math.sqrt(x ** 2)
    #     return abs(type(x)(n))

def lese_lab_datei(fuckdieamcas):

    f = open(fuckdieamcas)
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
    return np.genfromtxt(io.BytesIO(data.encode('utf-8')))

#dataR =  lese_lab_datei('rauschmessung.lab')
#zR  =  np.array([x[1]for x in dataR])
#dR =  np.array([x[2]for x in dataR])
#MR=np.mean(dR)
#stdR=np.std(dR)
# data1 =  lese_lab_datei('1.lab')
# t1  =  np.array([x[1]for x in data1])
# w1 =  np.array([x[2]for x in data1])
# data2 =  lese_lab_datei('2.lab')
# t2  =  np.array([x[1]for x in data2])
# w2 =  np.array([x[2]for x in data2])
# data3 =  lese_lab_datei('3.lab')
# t3  =  np.array([x[1]for x in data3])
# w3 =  np.array([x[2]for x in data3])
# data4 =  lese_lab_datei('4.lab')
# t4  =  np.array([x[1]for x in data4])
# w4 =  np.array([x[2]for x in data4])
# data5 =  lese_lab_datei('5.lab')
# t5  =  np.array([x[1]for x in data5])
# w5 =  np.array([x[2]for x in data5])
data6 =  lese_lab_datei('6.lab')
t6  =  np.array([x[1]for x in data6])
w6 =  np.array([x[2]for x in data6])
# data7 =  lese_lab_datei('7.lab')
# t7  =  np.array([x[1]for x in data7])
# w7 =  np.array([x[2]for x in data7])
# data8 =  lese_lab_datei('8.lab')
# t8  =  np.array([x[1]for x in data8])
# w8 =  np.array([x[2]for x in data8])
# data9 =  lese_lab_datei('9.lab')
# t9  =  np.array([x[1]for x in data9])
# w9 =  np.array([x[2]for x in data9])
# data10 =  lese_lab_datei('10.lab')
# t10  =  np.array([x[1]for x in data10])
# w10 =  np.array([x[2]for x in data10])
# data11 =  lese_lab_datei('11.lab')
# t11  =  np.array([x[1]for x in data11])
# w11 =  np.array([x[2]for x in data11])
# data12 =  lese_lab_datei('12.lab')
# t12  =  np.array([x[1]for x in data12])
# w12 =  np.array([x[2]for x in data12])
# data13 =  lese_lab_datei('13.lab')
# t13  =  np.array([x[1]for x in data13])
# w13 =  np.array([x[2]for x in data13])
# data14 =  lese_lab_datei('14.lab')
# t14  =  np.array([x[1]for x in data14])
# w14=  np.array([x[2]for x in data14])
# data15 =  lese_lab_datei('15.lab')
# t15  =  np.array([x[1]for x in data15])
# w15 =  np.array([x[2]for x in data15])
# data16 =  lese_lab_datei('16.lab')
# t16  =  np.array([x[1]for x in data16])
# w16 =  np.array([x[2]for x in data16])
# data17 =  lese_lab_datei('17.lab')
# t17  =  np.array([x[1]for x in data17])
# w17 =  np.array([x[2]for x in data17])
# data18 =  lese_lab_datei('18.lab')
# t18 =  np.array([x[1]for x in data18])
# w18 =  np.array([x[2]for x in data18])
# data19 =  lese_lab_datei('19.lab')
# t19  =  np.array([x[1]for x in data19])
# w19 =  np.array([x[2]for x in data19])
# data20 =  lese_lab_datei('20.lab')
# t20  =  np.array([x[1]for x in data20])
# w20  =  np.array([x[2]for x in data20])
# data21=  lese_lab_datei('21.lab')
# t21  =  np.array([x[1]for x in data21])
# w21 =  np.array([x[2]for x in data21])
# data22 =  lese_lab_datei('22.lab')
# t22  =  np.array([x[1]for x in data22])
# w22 =  np.array([x[2]for x in data22])
# data23 =  lese_lab_datei('23.lab')
# t23  =  np.array([x[1]for x in data23])
# w23 =  np.array([x[2]for x in data23])
# data24 =  lese_lab_datei('24.lab')
# t24  =  np.array([x[1]for x in data24])
# w24 =  np.array([x[2]for x in data24])
# data25 =  lese_lab_datei('25.lab')
# t25  =  np.array([x[1]for x in data25])
# w25 =  np.array([x[2]for x in data25])
# data26 =  lese_lab_datei('26.lab')
# t26  =  np.array([x[1]for x in data26])
# w26=  np.array([x[2]for x in data26])
# data27 =  lese_lab_datei('27.lab')
# t27  =  np.array([x[1]for x in data27])
# w27 =  np.array([x[2]for x in data27])
# data28 =  lese_lab_datei('28.lab')
# t28  =  np.array([x[1]for x in data28])
# w28 =  np.array([x[2]for x in data28])
# data29 =  lese_lab_datei('29.lab')
# t29  =  np.array([x[1]for x in data29])
# w29 =  np.array([x[2]for x in data29])
# data30 =  lese_lab_datei('30.lab')
# t30  =  np.array([x[1]for x in data30])
# w30 =  np.array([x[2]for x in data30])
# data31 =  lese_lab_datei('31.lab')
# t31  =  np.array([x[1]for x in data31])
# w31 =  np.array([x[2]for x in data31])

# w=[w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31]
# t=[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31]
#for i in range(31):
#    plt.plot(t[i],w[i],'r-')
 #   plt.grid()
  #  plt.axis([0,8,-21, 21])
   # plt.show()

#%%
plt.plot(t6,w6-np.mean(w6),color='blue', marker=',')
plt.grid()
plt.axis([0,8,-3,3])
plt.xlabel('Zeit/s')
plt.ylabel('Druck/hPa')
plt.title('Große Flasche')
roh1 = plt.gcf()
plt.show()
plt.draw()
roh1.savefig('grFlasche.eps', dpi=1024)

# plt.plot(t8,w8-np.mean(w8),color='red', marker=',')
# plt.grid()
# plt.axis([0,8,-12,12])
# plt.xlabel('Zeit/s')
# plt.ylabel('Druck/hPa')
# plt.title('Mittlere Flasche')
# roh2 = plt.gcf()
# plt.show()
# plt.draw()
# roh2.savefig('miFlasche.eps', dpi=1024)
#
# plt.plot(t28,w28-np.mean(w28),color='green', marker=',')
# plt.grid()
# plt.axis([0,8,-15,17.5 ])
# plt.xlabel('Zeit/s')
# plt.ylabel('Druck/hPa')
# plt.title('Kleine Flasche')
# roh3 = plt.gcf()
# plt.show()
# plt.draw()
# roh3.savefig('klFlasche.eps', dpi=1024)

#%%
freq, amp = fourier_fft(np.array(w6),np.array(t6))
fig,ax = plt.subplots()
ax.plot(freq,amp)
plt.show()
#%%
# t = np.arange(len(w24))
# huans =np.fft.fft(w24)
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, huans.real, freq, huans.imag)
# plt.show()

#%% internet

def fourier_transform(t, fkt):
    """
    Calculates the Fourier-Transformation of fkt with FFT.
    The output Fkt of the Fourier-Transformation of fkt
        is correctly normed (multiply with dt)
        is correctly ordered (swap first and second half of output-array).
    Calculates the frequencies from t.
    """
    n = t.size
    dt = t[1]-t[0]

    f=np.linspace(-1/(2*dt), 1/(2*dt), t.size, endpoint=False)
    Fkt = np.fft.fft(fkt)*dt

    # swap first and second half of output-array
    Fkt2 = np.fft.fftshift(Fkt)
    plt.plot(f, Fkt2,'r+')
    plt.axis([-0.5,0.5,-100,100 ])
    for i in range(len(Fkt2)):
        if Fkt2[i]>20:
            print(Fkt2[i],f[i])

    #return f, Fkt2





# fourier_transform(t28, -w28)
