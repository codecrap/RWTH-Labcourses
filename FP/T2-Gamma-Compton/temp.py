#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:37:37 2019

@author: alex
"""
import uncertainties.unumpy as unp
from uncertainties import ufloat
import numpy as np

#delim = (min(mean2)-max(mean1))/2
#def ChtoE(ch):#
#    if ch<=delim:
#        E = a1 * ch + b1
#        #dE = np.sqrt((ch*da1)**2 + (a1*dch)**2 + db1**2)
#    else:
#        E = a2 * ch + b2
#        dE = np.sqrt((ch*da2)**2 + (a2*dch)**2 + db2**2)
#    return [E, dE] # in keV

#a = unp.uarray([100, 150, 200], [1, 3, 2])
#b = unp.uarray([30, 45, 62], [2, 1, 3])

#c = a*b
#print(c)


u = ufloat(1, 0.1, "u variable")  # Tag
v = ufloat(10, 0.1, "v variable")
w = ufloat(15, 0.1, "v variable")
sum_value = u+2*v+w**2
for (var, error) in sum_value.error_components().items():
	print("{}: {}".format(var.tag, error))

#print(np.sqrt(0.1**2+0.2**2))
num = 3.5
print(np.size(num))






