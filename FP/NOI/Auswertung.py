# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:29:11 2019

@author: assil
"""

import scipy
import scipy.fftpack
import scipy.odr
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt,sin,cos,log,exp,diag
import uncertainties.unumpy as unp
from uncertainties import ufloat
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


#High/Lowpass filters calibration
G=np.array([0.010566037735849057, 0.040754716981132075, 0.0908490566037736, 0.15943396226415094, 0.2443396226415094, 0.3433962264150943, 0.44528301886792454, 0.5433962264150943, 0.6349056603773585, 0.7122641509433962, 0.7764150943396226, 0.8273584905660377, 0.8745283018867924, 0.8971698113207547, 0.920754716981132, 0.9377358490566038, 0.9433962264150944, 0.9622641509433962, 0.9811320754716981, 0.9716981132075472, 0.9905660377358491, 0.9433962264150944, 0.9, 0.8952830188679246, 0.8820754716981132, 0.8632075471698113, 0.8490566037735849, 0.8443396226415094, 0.839622641509434, 0.8320754716981132, 0.8188679245283018, 0.8018867924528302, 0.780188679245283, 0.7594339622641509, 0.7405660377358491, 0.7264150943396226, 0.7169811320754716, 0.709433962264151, 0.6839622641509434, 0.6386792452830189, 0.6009433962264151, 0.5716981132075472, 0.5462264150943396, 0.5103773584905661, 0.4792452830188679, 0.4547169811320755, 0.43207547169811317, 0.4066037735849057, 0.3839622641509434, 0.36415094339622645, 0.24245283018867925, 0.14622641509433962, 0.11037735849056603, 0.08235849056603774, 0.06292452830188679, 0.04962264150943396, 0.04084905660377359, 0.034339622641509436, 0.029056603773584905, 0.022075471698113205, 0.01773584905660377, 0.015000000000000001, 0.01330188679245283])
err_G = G*0.001

Freq = 100*np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 4.0, 6.0, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0])

err_Freq = np.ones(len(Freq))

fig = plt.figure()
plt.errorbar(log(Freq),G,xerr=log(err_Freq),yerr=err_G,linestyle='None',marker='.',label='Data')
plt.axhline(1/sqrt(2),color='red',linestyle='--')
plt.axvline(6.8977,color='red',linestyle='--')
plt.axvline(4.6495,color='red',linestyle='--')
plt.xlabel('log(Frequenz) [log(Hz)]')
plt.ylabel('Gain')
plt.legend(loc='best')




#_________________________________________________________________________________________________
#R Abhaengigkeit
R = np.array([1.,10.,100.,1000.,10000.,100000.,1000000])
e_R = R*0.001
V = np.array([8.83e-13,8.40e-13,9.12e-13,1.03e-12,2.59e-12,2.36e-11,1.83e-10])
e_V = np.array([5.90667550e-15,4.41527744e-14,2.02019813e-14,3.24376475e-14,4.02756406e-14,6.04569068e-13,2.80182413e-12])#fehlerrechungn für e_V muss gemacht werden bzw fehler zu klein 
V1 = V-8.83e-13
a,ea,b,eb,chiq,cov =lineare_regression_xy(R,V-8.83*10**(-13),e_R,e_V)
#print a,ea,b,eb
fig3 = plt.figure()
plt.plot(log(R),log(V),linestyle='None',marker='o',label='Data')
plt.xlabel(r'$ln(R) [ln(\Omega)]$')
plt.ylabel(r'$ln(<V_J^2+V_N^2>) [V^2]$')
plt.legend(loc='best')


fig4,(ax1,ax2) = plt.subplots(2,1,figsize=(8.,5.))
ax1.errorbar(R,V1,xerr=e_R,yerr=e_V,linestyle='None',marker='.',label='Data')
ax1.plot(R,a*R+b,label=r'$<V_J^2>=a*R+b$')
ax1.legend(loc='best')
ax1.set_ylabel(r'$<V_J^2> [V^2]$')
ax1.text(700000,0.0,r'$\frac{\chi^2}{ndof}$=%.3f'%(chiq/5))
diff = V1 - (a*R+b)
e_diff = sqrt(e_R**2+(a*e_V)**2)

ax2.axhline(0,color='r')
ax2.errorbar(R,diff,yerr=e_diff,linestyle='None')
ax2.set_ylabel(r'Residuen $[V^2]$')
ax2.set_xlabel('Widerstand R [\Omega]')

#________________________________________________________________________________
#Bandbreite Abhaengigkeit Stickstofftemperatur
ENBW = np.array([258,355,784,1077,3554,9997,19774,33324,35543,107740,110961,111061])
e_ENBW = ENBW*0.04
VN=np.array([4.12969637e-14,5.28957485e-14,7.55038812e-14,1.16347786e-13,3.42366885e-13,8.88075606e-13,9.57765109e-13,2.25234309e-12,2.52735871e-12,7.79526180e-12,8.44476894e-12,8.43664410e-12])
e_VN = np.array([2.24385577e-15,3.57012674e-15,2.63579393e-15,1.26762936e-15,2.93624311e-15,1.31142444e-14,5.67173212e-15,1.28170750e-14,1.50405477e-14,1.40088021e-13,1.98183482e-13,5.36442500e-14])
VNJ=np.array([6.25613045e-14,7.63976142e-14,1.62488730e-13,2.19539976e-13,7.03271647e-13,1.74180099e-12,1.90813094e-12,5.96649996e-12,6.69339990e-12,1.53089723e-11,1.59098151e-11,1.63854823e-11])
e_VNJ = np.array([3.06968080e-16,2.71604867e-15,4.88289741e-15,5.51874933e-15,1.26746494e-14,1.24843671e-14,3.20339695e-14,7.95310153e-14,7.38898304e-14,1.78201249e-13,2.86175192e-13,3.81870985e-13])
VJ = VNJ - VN
e_VJ = sqrt(e_VN**2+e_VNJ**2)

a,ea,b,eb,chiq,cov =lineare_regression_xy(ENBW,VJ,e_ENBW,e_VJ)
#print a,ea,b,eb
fig5,(ax1,ax2) = plt.subplots(2,1,figsize=(8.,5.))
ax1.errorbar(ENBW,VJ,xerr=e_ENBW,yerr=e_VJ,linestyle='None',marker='.',label='Data')
ax1.plot(ENBW,a*ENBW+b,label=r'$<V_J^2>=a*\Delta f+b$')
ax1.legend(loc='best')
ax1.set_ylabel(r'$<V_J^2> [V^2]$')
ax1.text(100000,0.0,r'$\frac{\chi^2}{ndof}$=%.3f'%(chiq/10))
diff = VJ - (a*ENBW+b)
e_diff = sqrt(e_VJ**2+(a*e_ENBW)**2)

ax2.axhline(0,color='k')
ax2.errorbar(ENBW,diff,yerr=e_diff,linestyle='None')
ax2.set_ylabel(r'Residuen $[V^2]$')
ax2.set_xlabel('ENBW [Hz]')



#____________________________________________________________________________
#Bandbreite Abhaengigkeit Raumtemperatur
ENBW = np.array([258,355,784,1077,3554,9997,19774,33324,35543,107740,110961,111061])
e_ENBW = ENBW*0.04
VN = np.array([3.89501318e-14,4.34950022e-14,7.76223593e-14,1.15410914e-13,3.32656211e-13,8.82992590e-13,9.43262643e-13,2.28599479e-12,2.32196730e-12,7.80841836e-12,8.38072872e-12,8.03865816e-12])
e_VN = np.array([1.79771555e-15,2.05814002e-15,7.61088142e-16,1.41260829e-15,3.34045154e-15,1.28091413e-14,1.77646052e-14,4.21062855e-14,3.19479954e-14,7.56263819e-14,2.46546586e-13,1.19530564e-13])
VNJ = np.array([9.43312952e-14,1.21090700e-13,2.59301251e-13,3.75829344e-13,1.18724627e-12,2.60974735e-12,2.77916626e-12,1.13227273e-11,1.17245777e-11,2.94788470e-11,2.98919264e-11,3.05553569e-11])
e_VNJ = np.array([6.18962860e-15,4.09407194e-15,2.65278639e-15,5.36046504e-15,2.31303632e-14,2.21593514e-14,4.17207671e-14,2.42556286e-13,2.01330373e-13,6.15527881e-13,7.00759635e-13,3.36968049e-13])
VJ = VNJ - VN
e_VJ = sqrt(e_VN**2+e_VNJ**2)

a,ea,b,eb,chiq,cov =lineare_regression_xy(ENBW,VJ,e_ENBW,e_VJ)
#print a,ea,b,eb
fig6,(ax1,ax2) = plt.subplots(2,1,figsize=(8.,5.))
ax1.errorbar(ENBW,VJ,xerr=e_ENBW,yerr=e_VJ,linestyle='None',marker='.',label='Data')
ax1.plot(ENBW,a*ENBW+b,label=r'$<V_J^2>=a*\Delta f+b$')
ax1.legend(loc='best')
ax1.set_ylabel(r'$<V_J^2> [V^2]$')
ax1.text(100000,0.0,r'$\frac{\chi^2}{ndof}$=%.3f'%(chiq/10))
diff = VJ - (a*ENBW+b)
e_diff = sqrt(e_VJ**2+(a*e_ENBW)**2)

ax2.axhline(0,color='k')
ax2.errorbar(ENBW,diff,yerr=e_diff,linestyle='None')
ax2.set_ylabel(r'Residuen $[V^2]$')
ax2.set_xlabel('ENBW [Hz]')
