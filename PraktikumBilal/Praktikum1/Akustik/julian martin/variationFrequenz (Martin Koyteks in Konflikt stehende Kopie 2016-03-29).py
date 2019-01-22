# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:03:18 2016

@author: Defender833
"""

import numpy as np
import Praktikum as p
import matplotlib.pyplot as plt

messung_400 = np.zeros(9)
messung_800 = np.zeros(9)
messung_1200 = np.zeros(9)
messung_1600 = np.zeros(9)
messung_2000 = np.zeros(9)
messung_2400 = np.zeros(9)


messung_400[0] = 416.0
messung_400[1] = 404.9
messung_400[2] = 417.0
messung_400[3] = 410.7
messung_400[4] = 404.3
messung_400[5] = 423.0
messung_400[6] = 416.1
messung_400[7] = 405.0
messung_400[8] = 419.3

messung_800[0] = 798.2
messung_800[1] = 826.8
messung_800[2] = 822.5
messung_800[3] = 822.1
messung_800[4] = 797.5
messung_800[5] = 836.4
messung_800[6] = 800.1
messung_800[7] = 791.5
messung_800[8] = 820.1

messung_1200[0] = 1210.0
messung_1200[1] = 1190.0
messung_1200[2] = 1212.6
messung_1200[3] = 1210.9
messung_1200[4] = 1194.0
messung_1200[5] = 1216.5
messung_1200[6] = 1210.0
messung_1200[7] = 1223.0
messung_1200[8] = 1195.7

messung_1600[0] = 1608.0
messung_1600[1] = 1573.9
messung_1600[2] = 1629.6
messung_1600[3] = 1629.0
messung_1600[4] = 1612.3
messung_1600[5] = 1631.3
messung_1600[6] = 1608.0
messung_1600[7] = 1628.2
messung_1600[8] = 1612.0

messung_2000[0] = 2041.4
messung_2000[1] = 2029.3
messung_2000[2] = 2031.1
messung_2000[3] = 2037.2
messung_2000[4] = 2013.8
messung_2000[5] = 2047.8
messung_2000[6] = 2021.3
messung_2000[7] = 2029.3
messung_2000[8] = 2019.4

messung_2400[0] = 2425.5
messung_2400[1] = 2433.4
messung_2400[2] = 2443.7
messung_2400[3] = 2446.7
messung_2400[4] = 2421.3
messung_2400[5] = 2467.3
messung_2400[6] = 2425.5
messung_2400[7] = 2439.2
messung_2400[8] = 2433.4

i = 0
check = True
mean_400 = 0.

while check:
    if i < len(messung_400):
        mean_400 += messung_400[i]
        i += 1
    else:
        check = False
mean_400 /= len(messung_400)

i = 0
check = True
mean_800 = 0.

while check:
    if i < len(messung_800):
        mean_800 += messung_800[i]
        i += 1
    else:
        check = False
mean_800 /= len(messung_800)

i = 0
check = True
mean_1200 = 0.

while check:
    if i < len(messung_1200):
        mean_1200 += messung_1200[i]
        i += 1
    else:
        check = False
mean_1200 /= len(messung_1200)

i = 0
check = True
mean_1600 = 0.

while check:
    if i < len(messung_1600):
        mean_1600 += messung_1600[i]
        i += 1
    else:
        check = False
mean_1600 /= len(messung_1600)

i = 0
check = True
mean_2000 = 0.

while check:
    if i < len(messung_2000):
        mean_2000 += messung_2000[i]
        i += 1
    else:
        check = False
mean_2000 /= len(messung_2000)

i = 0
check = True
mean_2400 = 0.

while check:
    if i < len(messung_2400):
        mean_2400 += messung_2400[i]
        i += 1
    else:
        check = False
mean_2400 /= len(messung_2400)

# print mean_400, mean_800, mean_1200, mean_1600, mean_2000, mean_2400

i = 0
check = True
x = 0.

while check:
    if i < len(messung_400):
        x += (messung_400[i]-mean_400)**2
        i += 1
    else:
        check = False
sig_400 = np.sqrt((x/(len(messung_400)-1)))
# print mean_400, " +- ", sig_400

i = 0
check = True
x = 0.

while check:
    if i < len(messung_800):
        x += (messung_800[i]-mean_800)**2
        i += 1
    else:
        check = False
sig_800 = np.sqrt((x/(len(messung_800)-1)))
# print mean_800, " +- ", sig_800

i = 0
check = True
x = 0.

while check:
    if i < len(messung_1200):
        x += (messung_1200[i]-mean_1200)**2
        i += 1
    else:
        check = False
sig_1200 = np.sqrt((x/(len(messung_1200)-1)))
# print mean_1200, " +- ", sig_1200

i = 0
check = True
x = 0.

while check:
    if i < len(messung_1600):
        x += (messung_1600[i]-mean_1600)**2
        i += 1
    else:
        check = False
sig_1600 = np.sqrt((x/(len(messung_1600)-1)))
# print mean_1600, " +- ", sig_1600

i = 0
check = True
x = 0.

while check:
    if i < len(messung_2000):
        x += (messung_2000[i]-mean_2000)**2
        i += 1
    else:
        check = False
sig_2000 = np.sqrt((x/(len(messung_2000)-1)))
# print mean_2000, " +- ", sig_2000

i = 0
check = True
x = 0.

while check:
    if i < len(messung_2400):
        x += (messung_2400[i]-mean_2400)**2
        i += 1
    else:
        check = False
sig_2400 = np.sqrt((x/(len(messung_2400)-1)))
# print mean_2400, " +- ", sig_2400

''' Resonanz bei fester Rohrlänge '''
L = np.zeros(5)
L[0] = 0.425
L[1] = 0.424
L[2] = 0.425
L[3] = 0.425
L[4] = 0.425
sig_L = 0.001
mean_L = L.mean()
v_lit = 340

data = np.zeros(6)
data_sig = np.zeros(6)

data[0] = mean_400
data[1] = mean_800
data[2] = mean_1200
data[3] = mean_1600
data[4] = mean_2000
data[5] = mean_2400

data_sig[0] = sig_400
data_sig[1] = sig_800
data_sig[2] = sig_1200
data_sig[3] = sig_1600
data_sig[4] = sig_2000
data_sig[5] = sig_2400

n = np.ones(6)
n[0] = 1
n[1] = 2
n[2] = 3
n[3] = 4
n[4] = 5
n[5] = 6

# print data_sig 
# v = data * 2 * mean_L / n

lin_reg = p.lineare_regression(n, data, data_sig)
print lin_reg[0]*2*mean_L, " +- ", lin_reg[1]*2*mean_L
print lin_reg[4]/4

'''
x = np.linspace(0, 7, num=1000)
plt.xlabel("Frequenz", fontsize='large')
plt.ylabel("Nummer der Messung", fontsize='large')
plt.errorbar(n, data, yerr = data_sig, fmt=".")
plt.plot(x, lin_reg[0]*x + lin_reg[2], label='Lin_Reg')
plt.legend()
'''

''' Residuum '''

residuum = (data)-(lin_reg[0]*n+lin_reg[2])

plt.xlabel("Mittelwerte", fontsize='large')
plt.ylabel("Residuen",fontsize='large')
plt.errorbar(n, residuum, yerr=data_sig, fmt=".")
plt.hlines(0, 0, 7)
plt.legend()

