# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 14:12:12 2016

@author: Erik
"""

import numpy as np
import scipy as sp
import pylatex as py
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Package, Matrix
from pylatex.utils import italic
import os

"Daten Gruppe 1"
b=np.array([ 41.73686211 , 42.07569322 , 43.27057685 , 41.62381085 , 42.50562048 , 43.34321926 , 42.88345142 , 42.64868708 , 41.19405156 , 44.01308611 , 41.445079 , 41.19726736 , 42.88128366 , 45.11311522 , 39.82348889 , 40.41290307 , 41.56275801 , 38.77901745 , 42.02590667 , 36.24455829 , 37.06007198]).round(2)
a=np.array([366.36706036 , 363.81107315 , 361.64581498 , 359.7194054 , 358.00330334 , 356.39438868 , 354.94951955 , 353.53404422 , 352.273564 , 351.06223364 , 349.95063747 , 348.87088331 , 347.84145454 , 346.86412083 , 345.9082871 , 344.98884248 , 343.31592119 , 342.39623532 , 341.56764897 , 340.67836518 , 339.78027799]).round(2)
"statistische Fehler"
b1=np.array([ 0.34207061 , 0.32699637 , 0.33947823 , 0.32415161 , 0.3265512 ,  0.29422951 , 0.27177976 , 0.36542877 , 0.32655547 , 0.38440528 , 0.394984 , 0.2987074 , 0.40555263 , 0.4318308 , 0.44619882 , 0.41403767 , 0.1433718 , 0.1630226 , 0.21180899 , 0.17453278 , 0.18934393]).round(3)
"systematische Fehler"
b2=np.array([ 1.043 , 1.052 , 1.082 , 1.041 , 1.063 , 1.084 , 1.073 , 1.067 , 1.031 , 1.102 , 1.038 , 1.032 , 1.075 , 1.131 , 0.999 , 1.014 , 1.044 , 0.975 , 1.057 , 0.914 , 0.935])

if __name__ == '__main__':
    doc = Document()
    doc.packages.append(Package('geometry', options=['left=2.5cm','right=2.5cm','top=2cm','bottom=2cm']))
    with doc.create(Tabular('c|c|c|c|c')) as table:
        table.add_row(('Abschnitt','T in K', 'lambda in ','stat','sys'))
        table.add_hline()
        table.add_row((1,a[0], b[0],b1[0],b2[0]))
        table.add_row((2,a[1], b[1],b1[1],b2[1]))
        table.add_row((3,a[2], b[2],b1[2],b2[2]))
        table.add_row((4,a[3], b[3],b1[3],b2[3]))
        table.add_row((5,a[4], b[4],b1[4],b2[4]))
        table.add_row((6,a[5], b[5],b1[5],b2[5]))
        table.add_row((7,a[6], b[6],b1[6],b2[6]))
        table.add_row((8,a[7], b[7],b1[7],b2[7]))
        table.add_row((9,a[8], b[8],b1[8],b2[8]))
        table.add_row((10,a[9], b[9],b1[9],b2[9]))
        table.add_row((11, a[10], b[10],b1[10],b2[10]))
        table.add_row((12, a[11], b[11],b1[11],b2[11]))
        table.add_row((13, a[12], b[12],b1[12],b2[12]))
        table.add_row(( 14,a[13], b[13],b1[13],b2[13]))
        table.add_row(( 15,a[14], b[14],b1[14],b2[14]))
        table.add_row(( 16,a[15], b[15],b1[15],b2[15]))
        table.add_row(( 17,a[16], b[16],b1[16],b2[16]))
        table.add_row(( 18,a[17], b[17],b1[17],b2[17]))
        table.add_row(( 19,a[18], b[18],b1[18],b2[18]))
        table.add_row(( 20,a[19], b[19],b1[19],b2[19]))
        table.add_row(( 21,a[20], b[20],b1[20],b2[20]))
doc.generate_tex('Tabelle_TLambda_G1')

#%%
"Daten Gruppe 2"
d=np.array([42.18040802 , 41.16951297 , 41.95958525 , 40.96897788 , 41.80498775 , 42.23595663 , 42.31141954 , 43.03159653 , 42.78621412 , 40.84383049]).round(2)
c=np.array([367.93208757 , 364.12930851 , 360.76011746 , 357.71095481 , 355.02551687 , 352.60207685 , 350.38310018 , 348.40318132 , 346.54233391 , 344.82860945]).round(2)
d1=np.array([ 0.27250185 , 0.15645391 , 0.10231621,  0.10009293,  0.11976347,  0.11748202,  0.13574482 , 0.14125986 , 0.16178094 , 0.1748259 , 0.18828452 , 0.18236064,  0.14585534 , 0.12049352 , 0.09870311 , 0.08897636]).round(3)
d2=np.array([ 0.008 , 0.017 , 0.026 , 0.036 , 0.045 , 0.055 , 0.065 , 0.074 , 0.084 , 0.094])

if __name__ == '__main__':
    doc = Document()
    doc.packages.append(Package('geometry', options=['left=2.5cm','right=2.5cm','top=2cm','bottom=2cm']))
    with doc.create(Tabular('c|c|c|c|c')) as table:
        table.add_row(('Abschnitt','T in K', 'lambda in ','stat','sys'))
        table.add_hline()
        table.add_row((1,c[0], d[0],d1[0],d2[0]))
        table.add_row((2,c[1], d[1],d1[1],d2[1]))
        table.add_row((3,c[2], d[2],d1[2],d2[2]))
        table.add_row((4,c[3], d[3],d1[3],d2[3]))
        table.add_row((5,c[4], d[4],d1[4],d2[4]))
        table.add_row((6,c[5], d[5],d1[5],d2[5]))
        table.add_row((7,c[6], d[6],d1[6],d2[6]))
        table.add_row((8,c[7], d[7],d1[7],d2[7]))
        table.add_row((9,c[8], d[8],d1[8],d2[8]))
        table.add_row((10,c[9], d[9],d1[9],d2[9]))
doc.generate_tex('Tabelle_TLambda_G2')





