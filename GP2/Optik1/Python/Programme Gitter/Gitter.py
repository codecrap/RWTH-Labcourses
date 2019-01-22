# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:18:12 2017

@author: Jonathan
"""

import Praktikumsroutinen_DW
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import matplotlib.patches as mpatches

import scipy.optimize as sp
import sys
sys.path.append("../../../PraktikumPyLib/")
import PraktLib as pl

def gew_mittel(x, ex):
    mean = np.sum(x/ex**2)/np.sum(1./ex**2)
    sigma = np.sqrt(1./np.sum(1./ex**2))
    return (mean, sigma)

def bminToDeg(value):
    return value/60.

def degToSr (value):
    return value*(2.*np.pi/360.)

def srToDeg (value):
    return value*(360./(2.*np.pi))

#Bestimmung des Nullinien-Winkels

phi_0_list = np.array([260.5, 260.0, 260.5, 260.0, 260.5, 260.5, 260.5, 260.5, 260.5, 260.5])
phi_0_bminlist = np.array([0, 25, 0, 29, 1, 0, 3, 2, 0, 1])
phi_0_added = degToSr(phi_0_list + bminToDeg(phi_0_bminlist))

phi_0 = np.mean(phi_0_added)
ephi_prov = np.std(phi_0_added)
ephi_0 = ephi_prov/np.sqrt(len(phi_0_added))

# Fehler auf alle weiteren Winkelmessungen:
ephi = ephi_prov/np.sqrt(3.)

# Bestimmung der Gitterkonstante

def combineDegs(constDeg, mins):
	return degToSr(constDeg + np.mean( bminToDeg( np.array(mins))))

#blau
b_p1 = combineDegs(277.0,[11, 10, 10])
b_p2 = combineDegs(295.5,[ 9, 10, 10])
b_m1 = combineDegs(243.5,[26, 25, 25])
b_m2 = combineDegs(225.5,[19, 18, 17])

#grün
g_p1 = combineDegs(278.5,[10, 11, 12])
g_p2 = combineDegs(299.0,[17, 18, 18])
g_m1 = combineDegs(242.0,[16, 26, 28])
g_m2 = combineDegs(222.0,[17, 18, 19])

#gelb (1)
y_p1 = combineDegs(281.0,[16, 15, 14])
y_p2 = combineDegs(305.5,[21, 20, 22])
y_m1 = combineDegs(239.5,[24, 23, 25])
y_m2 = combineDegs(215.5,[31, 27, 28])

#rot
r_p1 = combineDegs(283.5,[20, 19, 18])
r_p2 = combineDegs(313.0,[ 9, 10, 10])
r_m1 = combineDegs(237.0,[24, 22, 24])
r_m2 = combineDegs(209.0,[ 5,  3,  6])


print ("\phi_0 = %.5f \pm %.5f sr = %.4f \pm %.4f Grad" %(phi_0, ephi_0, srToDeg(phi_0), srToDeg(ephi_0)))
print ("- - - - - - - - - - - - - - - - - - - - - -")
print (" \hline Farbe & -2. Ordnung   &   -1. Ordnung  &   1.Ordnung     &   2. Ordnung \\")
print (" \hline blau $\lambda = 467.81 nm$    &   $%.3f $   &   $%.3f$    &   $%.3f$    &   $%.3f$ \\" %(b_m2, b_m1, b_p1, b_p2))
print (" \hline grün-blau $\lambda = 508.58 nm$    &   $%.3f $   &   $%.3f$    &   $%.3f$    &   $%.3f$ \\" %(g_m2, g_m1, g_p1, g_p2))
print (" \hline gelb (1) $\lambda = 576.96 nm$    &   $%.3f $   &   $%.3f$    &   $%.3f$    &   $%.3f$ \\" %(y_m2, y_m1, y_p1, y_p2))
print (" \hline rot $\lambda = 643.85 nm$    &   $%.3f $   &   $%.3f$    &   $%.3f$    &   $%.3f$ \\" %(r_m2, r_m1, r_p1, r_p2))
print ("")
print (" alle obigen Werte haben die Unsicherheit $%.6f $" %(ephi))
print ("- - - - - - - - - - - - - - - - - - - - - -")


# nach Ordnungen sortiert
lambda_list = np.array([467.81, 508.58, 576.96, 643.85])
n_lambda = []
k = 0
while k <= 3:
    n_lambda.append(-2*lambda_list[k])
    k+=1
k = 0
while k <= 3:
    n_lambda.append(-1*lambda_list[k])
    k+=1
k = 0
while k <= 3:
    n_lambda.append(1*lambda_list[k])
    k+=1
k = 0
while k <= 3:
    n_lambda.append(2*lambda_list[k])
    k+=1

n_lambda = np.array(n_lambda)

# nach Ordnung sortiert
angle_list = np.array([b_m2, g_m2, y_m2, r_m2, b_m1, g_m1, y_m1, r_m1, b_p1, g_p1, y_p1, r_p1, b_p2, g_p2, y_p2, r_p2])
phi_0_array = np.ones(len(angle_list))*phi_0
angle_list = angle_list - phi_0_array
sin_list = np.sin(angle_list)
esin_list = np.cos(angle_list)*ephi # Fehler auf phi_0 ist systematisch



# Anpassung an sin(phi) = 1/d * n*lambda
a, ea_stat, b, eb_stat, chi2 = Praktikumsroutinen_DW.residuen(n_lambda, sin_list, 0, esin_list, "nm", "", r"$n \cdot \lambda$ " ,r"$ sin( \theta) $", title = "Regression ohne Verdrehung", k = n_lambda[9], l = sin_list[0], o = n_lambda[6], p = 0.003)

# systematischen Fehler
sin_list_m = np.sin(angle_list - ephi_0)
sin_list_p = np.sin(angle_list + ephi_0)
a_m, ea_m, b_m, eb_m, chi2_m, corr_m = Praktikumsroutinen_DW.lineare_regression(n_lambda, sin_list_m, esin_list)
a_p, ea_p, b_p, eb_p, chi2_p, corr_p = Praktikumsroutinen_DW.lineare_regression(n_lambda, sin_list_p, esin_list)
ea_sys = np.abs(a_m - a_p)/2.
eb_sys = np.abs(b_m - b_p)/2.

# Gesamtfehler
ea = np.sqrt(ea_stat**2 + ea_sys**2)

# d und seine Fehler
d = 1/a
ed_stat = ea_stat/a**2
ed_sys = ea_sys/a**2
ed = np.sqrt(ed_stat**2 + ed_sys**2)


a_real = 600./10**6
d_real = 1./a_real
print ("ohne Beachtung der Verdrehung:")
print (" a = (%.3f \pm %.3f \pm %.3f) 1/mm, Abweichung: %.2f" %(a*10**6, ea_stat*10**6, ea_sys*10**6, np.abs(a - a_real)/ea))
print (" d = (%.3f \pm %.3f \pm %.3f) nm, Abweichung: %.2f" %(d, ed_stat, ed_sys, np.abs(d - d_real)/ed))
print (" b = (%.6f \pm %.6f)" %(b, eb_stat))
print (" chi2/ndof = %.2f" %(chi2))
print ("====================================================")


# Bestimmung des Verdrehungswinkels

f = lambda p,phi: (np.sin(p[0])+np.sin(phi-p[0]))
diff = lambda p,phi,y: ((y-f(p,phi))/ephi)**2

start_param = [0]
opt_param, cov, info, message, status = sp.leastsq(diff,start_param,args=(angle_list,n_lambda/d),full_output=True)
print("Verdrehung mit leastsq: ",srToDeg(opt_param),cov,info,message,status)
print("chiq/ndf = ", pl.chiq(f(opt_param,angle_list),n_lambda/d,yerrors=ephi,ndf=len(n_lambda)-len(start_param)))
print(n_lambda,len(n_lambda)==len(angle_list))
print()

fig,ax = plt.subplots()
ax.plot(srToDeg(angle_list),n_lambda/d,'r+',srToDeg(angle_list),f(opt_param,angle_list),'k--')

vals = Praktikumsroutinen_DW.residuen(n_lambda/d, f(opt_param,angle_list), 0,np.ones(len(angle_list))*ephi, "nm", "", r"$n da$ " , r"$ sinta) $", title = r" Grad$" , k = n_lambda[9], l = sin_list[0], o = - 200., p = -0.0007)

print("chiq/ndf = ",vals[-1])
plt.show()

def verdrehung(angle, angle_list):
    an_list = np.ones(16)*angle
    y = np.sin(an_list) + np.sin(angle_list - an_list)
    ey = esin_list
    return y, ey


def findAngle (alpha_left, alpha_right, durchlauf, angle_list):

    left = alpha_left
    right = alpha_right
    mid = (right - left)/2. + left

    if durchlauf >= 20:
        a, ea_stat, b, eb_stat, chi2 = Praktikumsroutinen_DW.residuen(n_lambda, verdrehung(mid, angle_list)[0], 0, verdrehung(mid, angle_list)[1], "nm", "", r"$n \cdot \lambda$ " , r"$ sin(\varphi) + sin( \varphi - \theta) $", title = r"Regression mit Verdrehung um $\varphi = %.4f Grad$" %(srToDeg(mid)) , k = n_lambda[9], l = sin_list[0], o = - 200., p = -0.0007)
        # Systematik
        a_m, ea_m, b_m, eb_m, chi2_m, corr_m = Praktikumsroutinen_DW.lineare_regression(n_lambda, verdrehung(mid, angle_list - ephi_0)[0], verdrehung(mid, angle_list - ephi_0)[1])
        a_p, ea_p, b_p, eb_p, chi2_p, corr_p = Praktikumsroutinen_DW.lineare_regression(n_lambda, verdrehung(mid, angle_list + ephi_0)[0], verdrehung(mid, angle_list + ephi_0)[1])
        ea_sys = np.abs(a_m - a_p)/2.
        eb_sys = np.abs(b_m - b_p)/2.
        return mid, a, ea_stat, ea_sys, b, eb_stat, eb_sys, chi2
    else:
        alpha_r = right - (mid-left)/2.
        alpha_l = left + (mid-left)/2.
        y_l, ey_l = verdrehung(alpha_l, angle_list)
        a_l, ea_l, b_l, eb_l, chi_l, corr_r = Praktikumsroutinen_DW.lineare_regression(n_lambda, y_l, ey_l)

        y_r, ey_r = verdrehung(alpha_r, angle_list)
        a_r, ea_r, b_r, eb_r, chi_r, corr_r = Praktikumsroutinen_DW.lineare_regression(n_lambda, y_r, ey_r)

        if chi_r < chi_l:
            left = mid
        else:
            right = mid

        return findAngle(left, right, durchlauf + 1, angle_list)




turn, a, ea_stat, ea_sys, b, eb_stat, eb_sys, chi2 = findAngle(-np.pi/6., np.pi/6., 0, angle_list)
ea = np.sqrt(ea_sys**2 + ea_stat**2)
eb = np.sqrt(eb_sys**2 + ea_stat**2)

# Fehler auf Verdrehungswinkel turn aus Verschiebemethode angewandt auf Theta
angle_list_p = angle_list + np.sqrt(ephi**2 + ephi_0**2)   # kombinierter Fehler auf beide Winkel
angle_list_m = angle_list - np.sqrt(ephi**2 + ephi_0**2)

turn_m = findAngle(-np.pi/6., np.pi/6., 0, angle_list_m)[0]
turn_p = findAngle(-np.pi/6., np.pi/6., 0, angle_list_p)[0]

eturn = np.abs(turn_m - turn_p)/2.

# d und seine Fehler
d = 1/a
ed_stat = ea_stat/a**2
ed_sys = ea_sys/a**2
ed = np.sqrt(ed_stat**2 + ed_sys**2)

print ("mit Beachtung der Verdrehung:")
print (" Drehwinkel \phi = (%.8f \pm %.8f) sr = (%.6f \pm %.6f)°" %(turn, eturn, srToDeg(turn), srToDeg(eturn)))
print (" a = (%.3f \pm %.3f \pm %.3f) 1/mm, Abweichung: %.2f" %(a*10**6, ea_stat*10**6, ea_sys*10**6, np.abs(a - a_real)/ea))
print (" d = (%.3f \pm %.3f \pm %.3f) nm, Abweichung: %.2f" %(d, ed_stat, ed_sys, np.abs(d - d_real)/ed))
print (" b = (%.6f \pm %.6f)" %(b, eb_stat))
print (" chi2/ndof = %.2f" %(chi2))
print ("=============================================================")


#=======================================================================================
# Bestimmung der Wellenlängen

y1_m1 = combineDegs(239.0,[27, 28, 27]) - phi_0
y1_m2 = combineDegs(214.5,[18, 19, 18]) - phi_0
y1_p1 = combineDegs(281.5,[10, 10, 10]) - phi_0
y1_p2 = combineDegs(307.0,[ 2,  2,  1]) - phi_0

y2_m1 = combineDegs(239.0,[24, 25, 26]) - phi_0
y2_m2 = combineDegs(214.5,[13, 15, 13]) - phi_0
y2_p1 = combineDegs(281.5,[13, 13, 13]) - phi_0
y2_p2 = combineDegs(307.0,[ 5,  6,  5]) - phi_0


etheta = np.sqrt((ephi**2 + ephi_0**2))

def wavelength (theta, order):
    wl  = d*(np.sin(turn) + np.sin(theta - turn))/order
    ewl = 1./np.abs(order)*np.sqrt( (np.sin(turn) + np.sin(theta - turn))**2 * ed**2
                                  + d**2 * (np.cos(turn) - np.cos(theta - turn))**2 * eturn**2
                                  + d**2 * np.cos(theta - turn)**2 * etheta**2)
    return wl, ewl

# Wellenlängen der Einzelmessungen
lambda_list_1 = []
lambda_list_1.append(wavelength(y1_m2, -2.)[0])
lambda_list_1.append(wavelength(y1_m1, -1.)[0])
lambda_list_1.append(wavelength(y1_p1,  1.)[0])
lambda_list_1.append(wavelength(y1_p2,  2.)[0])
lambda_list_1 = np.array(lambda_list_1)

elambda_list_1 = []
elambda_list_1.append(wavelength(y1_m2, -2.)[1])
elambda_list_1.append(wavelength(y1_m1, -1.)[1])
elambda_list_1.append(wavelength(y1_p1,  1.)[1])
elambda_list_1.append(wavelength(y1_p2,  2.)[1])
elambda_list_1 = np.array(elambda_list_1)


lambda_list_2 = []
lambda_list_2.append(wavelength(y2_m2, -2.)[0])
lambda_list_2.append(wavelength(y2_m1, -1.)[0])
lambda_list_2.append(wavelength(y2_p1,  1.)[0])
lambda_list_2.append(wavelength(y2_p2,  2.)[0])
lambda_list_2 = np.array(lambda_list_2)

elambda_list_2 = []
elambda_list_2.append(wavelength(y2_m2, -2.)[1])
elambda_list_2.append(wavelength(y2_m1, -1.)[1])
elambda_list_2.append(wavelength(y2_p1,  1.)[1])
elambda_list_2.append(wavelength(y2_p2,  2.)[1])
elambda_list_2 = np.array(elambda_list_2)

lambda_1, elambda_1 = gew_mittel(lambda_list_1, elambda_list_1)
lambda_2, elambda_2 = gew_mittel(lambda_list_2, elambda_list_2)

# Augabe
print(" Ordnung in aufsteigender Reihenfolge ")
k = 0
while k <= 3:
    print (" Wellenlänge 1, Einzelmessung "+ str(k+1) + ": %.3f \pm %.3f nm" % (lambda_list_1[k], elambda_list_1[k]))
    k += 1

k = 0
while k <= 3:
    print (" Wellenlänge 2, Einzelmessung "+ str(k+1) + ": %.3f \pm %.3f nm" % (lambda_list_2[k], elambda_list_2[k]))
    k += 1

real_1 = 589.0
real_2 = 589.59

print (" Mittelwerte: ")
print (" Wellenlänge 1:  %.2f \pm %.2f nm, Abweichung: %.2f" % (lambda_1, elambda_1, np.abs(real_1 - lambda_1)/elambda_1))
print (" Wellenlänge 2:  %.2f \pm %.2f nm, Abweichung: %.2f" % (lambda_2, elambda_2, np.abs(real_2 - lambda_2)/elambda_2))
print ("==========================================================")

#===========================================================
# Auflösungsvermögen

a_m2 = 0.75*10**6
a_m1 = 1.75*10**6
a_p1 = 1.75*10**6
a_p2 = 0.75*10**6

ea = 0.5/np.sqrt(12)*10**6

A_m2 = 2 * a_m2/d
A_m1 = 1 * a_m1/d
A_p1 = 1 * a_p1/d
A_p2 = 2 * a_p2/d
eA_1 = np.sqrt(ea**2/d**2 + a**2/d**4 * ed**2)
eA_2 = 2 * np.sqrt(ea**2/d**2 + a**2/d**4 * ed**2)

A_min = real_1/(real_2 - real_1)

A_mean, eA_mean = gew_mittel(np.array([A_m2, A_m1, A_p1, A_p2]),np.array([eA_2, eA_1, eA_1, eA_2]) )

print("mindestens benötigtes Auflösungsvermögen: %.2f" %A_min)
print("gemessenes Auflösungsvermögen:")
print(" -2.Ordnung: %.10f \pm %.2f " %(A_m2, eA_2))
print(" -1.Ordnung: %.2f \pm %.2f " %(A_m1, eA_1))
print("  1.Ordnung: %.2f \pm %.2f " %(A_p1, eA_1))
print("  2.Ordnung: %.2f \pm %.2f " %(A_p2, eA_2))
print("  im Mittel: %.2f \pm %.2f " %(A_mean, eA_mean))
print(" Abweichung: %.4f" %(np.abs(A_mean-A_min)/eA_mean))
