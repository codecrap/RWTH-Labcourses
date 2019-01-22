# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:28:31 2016

@author: Erik
"""
import numpy as np
import scipy as sp
import pylatex as py
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Package, Matrix
from pylatex.utils import italic
import os

R=8.314472

"""
"T relative Fehler"
sig_lamb_T1 = sig_rel_T*lamb

"T absolute Fehler"
sig_lamb_T2 = sig_T*lamb/T

"P Relative Fehler"
sig_lamb_P1 = R*T*sig_rel_P

"P absolute Fehler"
sig_lamb_P2 = R*T*sig_P*np.exp(lamb/(R*T))
"""


"Daten Gruppe 1"
lamb=np.array([ 41.73686211 , 42.07569322 , 43.27057685 , 41.62381085 , 42.50562048 , 43.34321926 , 42.88345142 , 42.64868708 , 41.19405156 , 44.01308611 , 41.445079 , 41.19726736 , 42.88128366 , 45.11311522 , 39.82348889 , 40.41290307 , 41.56275801 , 38.77901745 , 42.02590667 , 36.24455829 , 37.06007198])*1000
T=np.array([366.36706036 , 363.81107315 , 361.64581498 , 359.7194054 , 358.00330334 , 356.39438868 , 354.94951955 , 353.53404422 , 352.273564 , 351.06223364 , 349.95063747 , 348.87088331 , 347.84145454 , 346.86412083 , 345.9082871 , 344.98884248 , 343.31592119 , 342.39623532 , 341.56764897 , 340.67836518 , 339.78027799])

"""Fehler auf P"""
"Skript"
sig_P_skript = 0.03 #Rel Fehler
sig_T_sensor = 2.5 #absolut
sig_T_konv = 0.01 #rel

"Kalibrierung"
sig_T_kal=0.0137 #absolut
print "Gruppe 1"

"sig_Lambdas"
#print"sig_lamb_skript= ",(sig_P_skript*R*T).round()/1000
#print"sig_lamb_sensor=", (sig_T_sensor*lamb/T).round()/1000
#print"sig_lamb_konv=", (sig_T_konv*lamb).round()/1000
"sig_ges_hersteller"
a=(np.sqrt((sig_P_skript*R*T)**2+(sig_T_sensor*lamb/T)**2+(sig_T_konv*lamb)**2)).round()/1000
b=(sig_T_kal*lamb/T).round()/1000
c=(np.sqrt((sig_P_skript*R*T)**2+(sig_T_sensor*lamb/T)**2+(sig_T_konv*lamb)**2+(sig_T_kal*lamb/T)**2)).round()/1000
#%%



if __name__ == '__main__':
    doc = Document()
    doc.packages.append(Package('geometry', options=['left=2.5cm','right=2.5cm','top=2cm','bottom=2cm']))
    with doc.create(Subsection('Table of something')):
        with doc.create(Tabular('c|c|c|c')) as table:
            table.add_row((' ', 'Hersteller', 'Kalibration', 'Gesamt'))
            table.add_hline()
            table.add_row(('Lambda0', a[0], b[0], c[0]))
            table.add_row(('Lambda1', a[1], b[1], c[1]))
            table.add_row(('Lambda2', a[2], b[2], c[2]))
            table.add_row(('Lambda3', a[3], b[3], c[3]))
            table.add_row(('Lambda4', a[4], b[4], c[4]))
            table.add_row(('Lambda5', a[5], b[5], c[5]))
            table.add_row(('Lambda6', a[6], b[6], c[6]))
            table.add_row(('Lambda7', a[7], b[7], c[7]))
            table.add_row(('Lambda8', a[8], b[8], c[8]))
            table.add_row(('Lambda9', a[9], b[9], c[9]))
            table.add_row(('Lambda10', a[10], b[10], c[10]))
            table.add_row(('Lambda11', a[11], b[11], c[11]))
            table.add_row(('Lambda12', a[12], b[12], c[12]))
            table.add_row(('Lambda13', a[13], b[13], c[13]))
            table.add_row(('Lambda14', a[14], b[14], c[14]))
            table.add_row(('Lambda15', a[15], b[15], c[15]))
            table.add_row(('Lambda16', a[16], b[16], c[16]))
            table.add_row(('Lambda17', a[17], b[17], c[17]))
            table.add_row(('Lambda18', a[18], b[18], c[18]))
            table.add_row(('Lambda19', a[19], b[19], c[19]))
            table.add_row(('Lambda20', a[20], b[20], c[20]))
doc.generate_tex('Tabelle_1')

#%%
"Daten Gruppe 2"
lamb=np.array([42.18040802 , 41.16951297 , 41.95958525 , 40.96897788 , 41.80498775 , 42.23595663 , 42.31141954 , 43.03159653 , 42.78621412 , 40.84383049])*1000
T=np.array([367.93208757 , 364.12930851 , 360.76011746 , 357.71095481 , 355.02551687 , 352.60207685 , 350.38310018 , 348.40318132 , 346.54233391 , 344.82860945])
print"Gruppe 2"
"sig_Lambdas"

#print"sig_lamb_skript= ",(sig_P_skript*R*T).round()/1000
#print"sig_lamb_sensor=", (sig_T_sensor*lamb/T).round()/1000
#print"sig_lamb_konv=", (sig_T_konv*lamb).round()/1000
d=(np.sqrt((sig_P_skript*R*T)**2+(sig_T_sensor*lamb/T)**2+(sig_T_konv*lamb)**2)).round()/1000
e=(sig_T_kal*lamb/T).round()/1000
f=(np.sqrt((sig_P_skript*R*T)**2+(sig_T_sensor*lamb/T)**2+(sig_T_konv*lamb)**2+(sig_T_kal*lamb/T)**2)).round()/1000

#%%
if __name__ == '__main__':
    doc = Document()
    doc.packages.append(Package('geometry', options=['left=2.5cm','right=2.5cm','top=2cm','bottom=2cm']))
    with doc.create(Subsection('Table of something')):
        with doc.create(Tabular('c|c|c|c')) as table:
            table.add_row((' ', 'Hersteller', 'Kalibration', 'Gesamt'))
            table.add_hline()
            table.add_row(('Lambda0', d[0], e[0], f[0]))
            table.add_row(('Lambda1', d[1], e[1], f[1]))
            table.add_row(('Lambda2', d[2], e[2], f[2]))
            table.add_row(('Lambda3', d[3], e[3], f[3]))
            table.add_row(('Lambda4', d[4], e[4], f[4]))
            table.add_row(('Lambda5', d[5], e[5], f[5]))
            table.add_row(('Lambda6', d[6], e[6], f[6]))
            table.add_row(('Lambda7', d[7], e[7], f[7]))
            table.add_row(('Lambda8', d[8], e[8], f[8]))
            table.add_row(('Lambda9', d[9], e[9], f[9]))
doc.generate_tex('Tabelle_sys_Fehler_G2')

