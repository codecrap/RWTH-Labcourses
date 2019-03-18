#-*- coding: utf-8 -*-
#
#@curve_plot.py:
#@author: Olexiy Fedorets
#@date: Thu 14.03.2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


# non-entangled photons
#
# Det1 fixed (to 90, max transmission), complete curve (-230..140)

vPolarizer0 = np.concatenate([np.arange(230,360,10),np.arange(0,150,10)])
vDet0 = [95386,115798,132592,143373,145887,144156,136960,122804,103533,80320,56503,
		 33772,16969,9715,13672,28934,60615,75903,100477,122279,138825,150336,153134,
		 152339,143444,128098,107448,83972]
vDet1 = [119642,120103,120576,121004,120703,120888,121104,121063,122215,122174,123095,
		 123021,122131,123173,123404,123114,123839,123487,124274,124131,124767,124362,
		 124879,125176,125693,125473,123377,123200]
vCoincidence = [5041,6227,7116,7673,7884,7902,7410,6461,5405,4062,2667,1365,506,102,
				332,1173,2362,3737,4982,6470,7436,7961,8152,8101,7524,6680,5477,4066]

fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0,'r-',
		np.linspace(0,280,vPolarizer0.size),vDet1,'b-',
		np.linspace(0,280,vPolarizer0.size),vCoincidence,'g-')

#
vPolarizerSet0 = [-45,0,45,90]
vPolarizerSet1 = [-23,23,68,113]
vAngles = zip(vPolarizerSet0,vPolarizerSet1)

# Det0 fixed to -45
vDetSet0 = [92347,92765,92171,92739]
vDetSet1 = [37671,34114,109552,112154]
vCoincidenceSet = [1347,931,4144,4474]

vDet0Continuous = [94448,94285,93878,94025,94276,93227,92773,92536,92092,90881,91008,
				   90471,90957,90576,90777,90252,90580,90160,90587,90005,90836,91475,91625,92265,92560,92388,92943,92691]
vDet1Continuous = [86188,100542,112565,119868,122684,119651,113318,101848,86942,68476,47814,
				   28388,15023,9070,12864,25595,43140,62890,82206,98173,110021,118311,121401,121926,114936,103966,88587,71785]
vCoincidenceContinuous = [3115,3953,4443,4895,5162,4819,4716,4205,3625,2934,1999,
						  1144,408,123,173,689,1486,2297,3088,3891,4374,4823,4877,4930,4604,4291,3639,2834]


fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0Continuous,'r-',
		np.linspace(0,280,vPolarizer0.size),vDet1Continuous,'b-',
		np.linspace(0,280,vPolarizer0.size),vCoincidenceContinuous,'g-')


# Det0 fixed to 0
vDetSet0 = [10220,10360,10338,9860]
vDetSet1 = [38100,35077,112668,122262]
vCoincidenceSet = [125,86,86,104]

vPolarizer0 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])

vDet0Continuous = [10374,9353,8239,8268,8081,8136,8102,8004,7152,7632,7706,7963,8129,8724]
vDet1Continuous = [84619,112002,121728,113468,87431,49689,16809,13704,45737,85146,113990,125020,116659,90976]
vCoincidenceContinuous = [81,77,63,90,74,115,112,99,85,80,84,75,95,92]

fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0Continuous,'r-',
		np.linspace(0,280,vPolarizer0.size),vDet1Continuous,'b-',
		np.linspace(0,280,vPolarizer0.size),vCoincidenceContinuous,'g-')

# Det0 fixed to +45
vDetSet0 = [89955,89831,88527,96738]
vDetSet1 = [43859,39274,110354,112436]
vCoincidenceSet = [980,1136,4154,4550]

vPolarizer0 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])

vDet0Continuous = [91419,92564,90307,90241,89500,90202,89988,89626,89522,89510,89560,89386,89060,89666]
vDet1Continuous = [83994,112018,121386,112121,87305,50413,16158,13913,43009,80158,109959,121077,112548,89703]
vCoincidenceContinuous = [3508,4611,4965,4633,3351,1746,314,337,1807,3424,4575,5029,4485,3322]

fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0Continuous,'r-',
		np.linspace(0,280,vPolarizer0.size),vDet1Continuous,'b-',
		np.linspace(0,280,vPolarizer0.size),vCoincidenceContinuous,'g-')

# Det0 fixed to 90
vDetSet0 = [144813,145610,146358,146993]
vDetSet1 = [35382,31807,109815,111959]
vCoincidenceSet = [1934,1730,7411,7529]

vPolarizer0 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])

vDet0Continuous = [147583,147499,147003,147780,148443,147964,148817,148357,149243,148533,149214,149119,148956,149904]
vDet1Continuous = [84205,114387,121461,113113,89087,50167,17057,16165,46553,86500,114167,125926,117863,92626]
vCoincidenceContinuous = [5470,7339,8340,7627,5726,3006,551,448,2730,5508,7471,8225,7712,5768]

fig,ax = plt.subplots()
ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0Continuous,'r-',
		np.linspace(0,280,vPolarizer0.size),vDet1Continuous,'b-',
		np.linspace(0,280,vPolarizer0.size),vCoincidenceContinuous,'g-')


# entangled photons
#
# Det1 fixed (to 90, max transmission), complete curve (-230..140)

# vPolarizer0 = np.concatenate([np.arange(230,360,10),np.arange(0,150,10)])
# vDet0 = [95386,115798,132592,143373,145887,144156,136960,122804,103533,80320,56503,
# 		 33772,16969,9715,13672,28934,60615,75903,100477,122279,138825,150336,153134,
# 		 152339,143444,128098,107448,83972]
# vDet1 = [119642,120103,120576,121004,120703,120888,121104,121063,122215,122174,123095,
# 		 123021,122131,123173,123404,123114,123839,123487,124274,124131,124767,124362,
# 		 124879,125176,125693,125473,123377,123200]
# vCoincidence = [5041,6227,7116,7673,7884,7902,7410,6461,5405,4062,2667,1365,506,102,
# 				332,1173,2362,3737,4982,6470,7436,7961,8152,8101,7524,6680,5477,4066]
#
# fig,ax = plt.subplots()
# ax.plot(np.linspace(0,280,vPolarizer0.size),vDet0,'r-',
# 		np.linspace(0,280,vPolarizer0.size),vDet1,'b-',
# 		np.linspace(0,280,vPolarizer0.size),vCoincidence,'g-')





plt.show()