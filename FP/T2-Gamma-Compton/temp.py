#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:37:37 2019

@author: alex
"""
import uncertainties.unumpy as unp
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

a = unp.uarray([100, 150, 200], [1, 3, 2])
b = unp.uarray([30, 45, 62], [2, 1, 3])

c = a*b
#print(c)
print(a[1].n)






