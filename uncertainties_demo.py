#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:33:34 2019

@author: alex
"""

import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

def uarray_tag(val, sig, tag):
    return np.array([ufloat(val[i], sig[i], tag) for i in range(np.size(val))])

def split_error(x):
	if isinstance(x, UFloat):
		sys = np.sqrt(sum(error**2 for (var, error) in x.error_components().items() if var.tag == "sys"))
		stat = np.sqrt(x.std_dev**2 - sys**2)
		return stat, sys
	else:
		sys = np.zeros(np.size(x))
		stat = np.zeros(np.size(x))
		for i in range(np.size(x)):
			sys[i] = np.sqrt(sum(error**2 for (var, error) in x[i].error_components().items() if var.tag == "sys"))
			stat[i] = np.sqrt(x[i].std_dev**2 - sys[i]**2)
		return stat, sys


a = ufloat(3, 0.5)
print(a.nominal_value)
print(a.std_dev)

vA = unp.uarray([1,2,3], [0.1,0.2,0.3])
print(unp.nominal_values(vA))
print(unp.std_devs(vA))

vB = unp.uarray([4,5,6], [0.4,0.5,0.6])

vC = vA/vB**2





vX = uarray_tag(vV1, vV1Err, 'sys')
vY = uarray_tag(vV2, vV2Err, 'sys')
vG = vX/vY

vA = uarray_tag(vB, vBErr, 'stat')

vC = vG+vA**2

vStat, vSys = split_error(vC)



