# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:08:22 2015

@author: lars
"""

import matplotlib.pyplot as plt
import numpy as np

datei=np.genfromtxt("Frequenz_fehler1.csv",delimiter=';')

f=open("Frequenz_fehler1.csv","r")
"""
zeilen=f.readlines()
print zeilen[0]
print zeilen[1]
gross_20=zeilen[2:6]
print gross_20[:1]
"""
N=np.array(y[0] for y in f)

"""
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

plt.figure()
plt.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
"""