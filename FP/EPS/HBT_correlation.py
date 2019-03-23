#-*- coding: utf-8 -*-
#
#@HBT_correlation.py:
#@author: Olexiy Fedorets
#@date: Tue 19.03.2019


import matplotlib.pyplot as plt
import numpy as np

vI = [20, ]

vt, v0, v1, v2, v01, v02, v12, v012 = np.genfromtxt('Data/HBT_correlation_45mA.quCNTPlot', skip_header=1)