#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:32:04 2019

@author: alex
"""

import matplotlib.pyplot as plt
import numpy as np

# set style for matplotlib
import matplotlib
matplotlib.style.use("../labreport.mplstyle")

# path needed for PraktLib
import sys
sys.path.append("./../../")

# import PraktLib take care of changes in module by manually reloading
import PraktLib as pl
from importlib import reload
pl = reload(pl)


# set strings and choose file
DATAPATH = "./Gwyddion/Gold/"
#FILE = "GAIN_opt_profile.txt"
#FILE = "GAIN_slow_profile.txt"
#FILE = "TIME_01_profile.txt"
FILE = "TIME_005_profile.txt"

# read data
with open(DATAPATH + FILE) as data:
		   vForwardX, vForwardY, vBackwardX, vBackwardY = np.genfromtxt((line.replace(',', '.') for line in data), skip_header=3).T


diff = np.abs(vForwardX[np.argmin(vForwardY)] - vBackwardX[np.argmin(vBackwardY)])
print(diff)