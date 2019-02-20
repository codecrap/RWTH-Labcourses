#-*- coding: utf-8 -*-
#
#@activity.py: compute lengths needed for compton scattering experiment setup
#@author: Olexiy Fedorets
#@date: Tue 19.02.2019


import numpy as np
import sys
sys.path.append("./../../")
import PraktLib as pl

# rD = abs
# rS = abs
d_large = (0.25 - 0.221)/2 + 0.221
d_medium = (0.199 - 0.171)/2 + 0.171
d_small = (0.149 - 0.121)/2 + 0.121
d_tiny = 0.085
# rD_long = np.mean([0.231,0.232,0.233,0.231])
# rD_short = np.mean([0.219,0.218,0.218,0.217])
rD_left = (0.23-0.211)/2 + 0.211
rD_right = (0.28-0.216)/2 + 0.216
rD = np.mean([rD_left,rD_right])

vTheta = pl.degToSr(np.array([50,40,30,24,19]))
# print(vTheta)

# alpha =  np.arctan()

rS = d_small/2 * np.tan(np.pi-np.arctan(2*rD/d_small) - vTheta)
print(rS)
