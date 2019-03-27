#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:33:34 2019

@author: alex
"""

import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

a = ufloat(3, 0.5)
print(a.nominal_value)
print(a.std_dev)

vA = unp.uarray([1,2,3], [0.1,0.2,0.3])
print(unp.nominal_values(vA))
print(unp.std_devs(vA))

vB = unp.uarray([4,5,6], [0.4,0.5,0.6])

vC = vA/vB**2

