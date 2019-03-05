#-*- coding: utf-8 -*-
#
#@activity.py: compute lengths needed for compton scattering experiment setup
#@author: Olexiy Fedorets
#@date: Tue 19.02.2019


import numpy as np
import sys
sys.path.append("./../../")
import PraktLib as pl
import uncertainties.unumpy as unp
from uncertainties import ufloat

# take care of changes in module by manually reloading
from importlib import reload
pl = reload(pl)

MEAS_ERROR = 0.001																	# 1mm tape accuracy

# ring diameters
d_large = ufloat((0.25 - 0.221)/2 + 0.221, MEAS_ERROR)
d_medium = ufloat((0.199 - 0.171)/2 + 0.171, MEAS_ERROR)
d_small = ufloat((0.149 - 0.121)/2 + 0.121, MEAS_ERROR)
d_tiny = ufloat(0.085, MEAS_ERROR)
vD_used = np.array([d_large,d_large,d_medium,d_small,d_small]) # @FIXME which rings were used exactly?


# distances to detector
# rD_long = np.mean([0.231,0.232,0.233,0.231])
# rD_short = np.mean([0.219,0.218,0.218,0.217])
rD_left = ufloat((0.23-0.211)/2 + 0.211, MEAS_ERROR)
rD_right = ufloat((0.28-0.216)/2 + 0.216, MEAS_ERROR)
rD = np.mean([rD_left,rD_right])

vTheta_required = pl.degToSr(np.array([50, 40, 30, 24, 19]))
print("Theta angles needed: \n",pl.srToDeg(vTheta_required))

# distances to source
vRs_required = vD_used/2 * unp.tan(np.pi - unp.arctan(2*rD/vD_used) - vTheta_required)
print("Distances to source needed: \n",vRs_required)


# now the other way round: get errors on theta angles through set rS distance
# assume distance is set to cm precision of requirement

vRs_set = np.array([ ufloat(np.round(x.nominal_value,2), 0.01) for _,x in enumerate(vRs_required)])
print("Distances to source set: \n",vRs_set)
vTheta_set = np.pi - unp.tan(2*vRs_set/vD_used) - unp.tan(2*rD/vD_used)
print("Theta angles set: \n",pl.srToDeg(vTheta_set))	#@FIXME this is bullshit results