#-*- coding: utf-8 -*-
#
#@nonentangled.py:
#@author: Olexiy Fedorets
#@date: Thu 14.03.2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import uncertainties.unumpy as unp

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

# non-entangled photons
########################
fig,ax = plt.subplots(3,2,figsize=(30,20))
fig.delaxes(ax[2,1])

# vPolarizerSet0 = [-45,0,45,90]
# vAngles = zip(vPolarizerSet0,vPolarizerSet1)

def f_cos(theta,a,b,c,d):
	return a * unp.cos( b * pl.degToSr(theta) + pl.degToSr(c) ) + d

# def f_cos(p,theta):
# 	return p[0] * np.cos( p[1] * pl.degToSr(theta) + pl.degToSr(p[2]) ) + p[2]

def polDeg(Nmin,Nmax):
	return Nmax/(Nmax+Nmin)
vPolDeg = []

def falseCoin(N0,N1):
	return np.array(N0) * np.array(N1) * 30e-9
mFalseCoin = []

########################
# Det0 fixed to -45
vPolarizer1Set = [-23,23,68,113]
vDet0Set = [92347,92765,92171,92739]
vDet1Set = [37671,34114,109552,112154]
vCoincidenceSet = [1347,931,4144,4474]

# actual degrees on polarizer
# vPolarizer1 = np.concatenate([-np.arange(230,360,10),np.arange(0,150,10)])
vPolarizer1 = np.arange(-130,150,10)
vDet0 = [94448,94285,93878,94025,94276,93227,92773,92536,92092,90881,91008,90471,
		 90957,90576,90777,90252,90580,90160,90587,90005,90836,91475,91625,92265,
		 92560,92388,92943,92691]
vDet1 = [86188,100542,112565,119868,122684,119651,113318,101848,86942,68476,47814,
		28388,15023,9070,12864,25595,43140,62890,82206,98173,110021,118311,121401,
		 121926,114936,103966,88587,71785]
vCoincidence = [3115,3953,4443,4895,5162,4819,4716,4205,3625,2934,1999,1144,408,123,
				173,689,1486,2297,3088,3891,4374,4823,4877,4930,4604,4291,3639,2834]

ax2 = ax[0,0].twinx()
ax[0,0].plot(vPolarizer1,vDet0,'rx',vPolarizer1Set,vDet0Set,'rs',label="Detektor 0")
ax[0,0].plot(vPolarizer1,vDet1,'bx',vPolarizer1Set,vDet1Set,'bs',label="Detektor 1")
ax2.plot(vPolarizer1,vCoincidence,'gx',vPolarizer1Set,vCoincidenceSet,'gs',label="Konzidenz 01")
ax[0,0].set_title("Polarisator 0 bei $-45^\circ$")
ax[0,0].set_xlabel("Polarisator 1 $(^\circ)$")
ax[0,0].set_ylabel("Detector counts")
ax2.set_ylabel("Koinzidenz counts")

# p0 = [-1e5,1,1,1e5]
# chifunc = lambda p,theta,y: (y-f_cos(theta,*p))
# optDict = spopt.least_squares(chifunc,p0,args=(vPolarizer1,vDet1),loss='cauchy',method='trf')


vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vCoincidence, p0=np.array([-1e5,1,1,1e5]))
ax2.plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'g-')
vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vDet1, p0=np.array([-1e5,1,1,1e5]))
vFiterror = np.sqrt(np.diag(mCov))
ax[0,0].plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'b-')

vUFit = unp.uarray(vFitparam,vFiterror)
Nmin = np.min(f_cos(vPolarizer1,*vUFit))
Nmax = np.max(f_cos(vPolarizer1,*vUFit))
vPolDeg += [polDeg(Nmin,Nmax)]

vFalseCoin += [falseCoin(vDet0,vDet1)]

########################
# Det0 fixed to 0
vPolarizer1Set = [-23,23,68,113]
vDet0Set = [10220,10360,10338,9860]
vDet1Set = [38100,35077,112668,122262]
vCoincidenceSet = [125,86,86,104]

# actual degrees on polarizer
# vPolarizer1 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])
vPolarizer1 = np.arange(-130,140,20)
vDet0 = [10374,9353,8239,8268,8081,8136,8102,8004,7152,7632,7706,7963,8129,8724]
vDet1 = [84619,112002,121728,113468,87431,49689,16809,13704,45737,85146,113990,125020,116659,90976]
vCoincidence = [81,77,63,90,74,115,112,99,85,80,84,75,95,92]

ax2 = ax[0,1].twinx()
ax[0,1].plot(vPolarizer1,vDet0,'rx',vPolarizer1Set,vDet0Set,'rs',label="Detektor 0")
ax[0,1].plot(vPolarizer1,vDet1,'bx',vPolarizer1Set,vDet1Set,'bs',label="Detektor 1")
ax2.plot(vPolarizer1,vCoincidence,'gx',vPolarizer1Set,vCoincidenceSet,'gs',label="Konzidenz 01")
ax[0,1].set_title("Polarisator 0 bei $0^\circ$")
ax[0,1].set_xlabel("Polarisator 1 $(^\circ)$")
ax[0,1].set_ylabel("Detector counts")
ax2.set_ylabel("Koinzidenz counts")

vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vCoincidence, p0=np.array([-1e5,1,1,1e5]))
ax2.plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'g-')
vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vDet1, p0=np.array([-1e5,1,1,1e5]))
vFiterror = np.sqrt(np.diag(mCov))
ax[0,1].plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'b-')

vUFit = unp.uarray(vFitparam,vFiterror)
Nmin = np.min(f_cos(vPolarizer1,*vUFit))
Nmax = np.max(f_cos(vPolarizer1,*vUFit))
vPolDeg += [polDeg(Nmin,Nmax)]

vFalseCoin += [falseCoin(vDet0,vDet1)]


########################
# Det0 fixed to +45
vPolarizer1Set = [-23,23,68,113]
vDet0Set = [89955,89831,88527,96738]
vDet1Set = [43859,39274,110354,112436]
vCoincidenceSet = [980,1136,4154,4550]

# actual degrees on polarizer
# vPolarizer1 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])
vPolarizer1 = np.arange(-130,140,20)
vDet0 = [91419,92564,90307,90241,89500,90202,89988,89626,89522,89510,89560,89386,89060,89666]
vDet1 = [83994,112018,121386,112121,87305,50413,16158,13913,43009,80158,109959,121077,112548,89703]
vCoincidence = [3508,4611,4965,4633,3351,1746,314,337,1807,3424,4575,5029,4485,3322]

ax2 = ax[1,0].twinx()
ax[1,0].plot(vPolarizer1,vDet0,'rx',vPolarizer1Set,vDet0Set,'rs',label="Detektor 0")
ax[1,0].plot(vPolarizer1,vDet1,'bx',vPolarizer1Set,vDet1Set,'bs',label="Detektor 1")
ax2.plot(vPolarizer1,vCoincidence,'gx',vPolarizer1Set,vCoincidenceSet,'gs',label="Konzidenz 01")
ax[1,0].set_title("Polarisator 0 bei $+45^\circ$")
ax[1,0].set_xlabel("Polarisator 1 $(^\circ)$")
ax[1,0].set_ylabel("Detector counts")
ax2.set_ylabel("Koinzidenz counts")

vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vCoincidence, p0=np.array([-1e5,1,1,1e5]))
ax2.plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'g-')
vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vDet1, p0=np.array([-1e5,1,1,1e5]))
vFiterror = np.sqrt(np.diag(mCov))
ax[1,0].plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'b-')

vUFit = unp.uarray(vFitparam,vFiterror)
Nmin = np.min(f_cos(vPolarizer1,*vUFit))
Nmax = np.max(f_cos(vPolarizer1,*vUFit))
vPolDeg += [polDeg(Nmin,Nmax)]

vFalseCoin += [falseCoin(vDet0,vDet1)]


########################
# Det0 fixed to 90
vPolarizer1Set = [-23,23,68,113]
vDet0Set = [144813,145610,146358,146993]
vDet1Set = [35382,31807,109815,111959]
vCoincidenceSet = [1934,1730,7411,7529]

# actual degrees on polarizer
# vPolarizer1 = np.concatenate([np.arange(230,360,20),np.arange(10,140,20)])
vPolarizer1 = np.arange(-130,140,20)
vDet0 = [147583,147499,147003,147780,148443,147964,148817,148357,149243,148533,149214,149119,148956,149904]
vDet1 = [84205,114387,121461,113113,89087,50167,17057,16165,46553,86500,114167,125926,117863,92626]
vCoincidence = [5470,7339,8340,7627,5726,3006,551,448,2730,5508,7471,8225,7712,5768]

ax2 = ax[1,1].twinx()
ax[1,1].plot(vPolarizer1,vDet0,'rx',vPolarizer1Set,vDet0Set,'rs',label="Detektor 0")
ax[1,1].plot(vPolarizer1,vDet1,'bx',vPolarizer1Set,vDet1Set,'bs',label="Detektor 1")
ax2.plot(vPolarizer1,vCoincidence,'gx',vPolarizer1Set,vCoincidenceSet,'gs',label="Konzidenz 01")
ax[1,1].set_title("Polarisator 0 bei $90^\circ$")
ax[1,1].set_xlabel("Polarisator 1 $(^\circ)$")
ax[1,1].set_ylabel("Detector counts")
ax2.set_ylabel("Koinzidenz counts")

vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vCoincidence, p0=np.array([-1e5,1,1,1e5]))
ax2.plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'g-')
vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer1, vDet1, p0=np.array([-1e5,1,1,1e5]))
vFiterror = np.sqrt(np.diag(mCov))
ax[1,1].plot(vPolarizer1,f_cos(vPolarizer1,*vFitparam),'b-')

vUFit = unp.uarray(vFitparam,vFiterror)
Nmin = np.min(f_cos(vPolarizer1,*vUFit))
Nmax = np.max(f_cos(vPolarizer1,*vUFit))
vPolDeg += [polDeg(Nmin,Nmax)]

vFalseCoin += [falseCoin(vDet0,vDet1)]


########################
# Det1 fixed (to 90, max transmission), complete curve (-230..140)

# actual degrees on polarizer
# vPolarizer0 = np.concatenate([-np.arange(230,360,10),np.arange(0,150,10)])
vPolarizer0 = np.arange(-130,150,10)
vDet0 = [95386,115798,132592,143373,145887,144156,136960,122804,103533,80320,56503,
		 33772,16969,9715,13672,28934,60615,75903,100477,122279,138825,150336,153134,
		 152339,143444,128098,107448,83972]
vDet1 = [119642,120103,120576,121004,120703,120888,121104,121063,122215,122174,123095,
		 123021,122131,123173,123404,123114,123839,123487,124274,124131,124767,124362,
		 124879,125176,125693,125473,123377,123200]
vCoincidence = [5041,6227,7116,7673,7884,7902,7410,6461,5405,4062,2667,1365,506,102,
				332,1173,2362,3737,4982,6470,7436,7961,8152,8101,7524,6680,5477,4066]

ax = plt.subplot(3,1,3)
ax2 = ax.twinx()
ax.plot(vPolarizer0,vDet0,'rx',label="Detektor 0")
ax.plot(vPolarizer0,vDet1,'bx',label="Detektor 1")
ax.plot(0,0,'g-x',label="Konzidenz 01")
ax2.plot(vPolarizer0,vCoincidence,'gx',label="Konzidenz 01")
ax.set_title("Polarisator 1 bei $90^\circ$ (max Transmission)")
ax.set_xlabel("Polarisator 0 $(^\circ)$")
ax.set_ylabel("Detector counts")
ax2.set_ylabel("Koinzidenz counts")
ax.legend(loc='best')

vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer0, vCoincidence, p0=np.array([-1e5,1,1,1e5]))
ax2.plot(vPolarizer0,f_cos(vPolarizer0,*vFitparam),'g-')
vFitparam, mCov = spopt.curve_fit(lambda theta,a,b,c,d: unp.nominal_values(f_cos(theta,a,b,c,d)), vPolarizer0, vDet0, p0=np.array([-1e5,1,1,1e5]))
vFiterror = np.sqrt(np.diag(mCov))
ax.plot(vPolarizer0,f_cos(vPolarizer0,*vFitparam),'r-')

vUFit = unp.uarray(vFitparam,vFiterror)
Nmin = np.min(f_cos(vPolarizer1,*vUFit))
Nmax = np.max(f_cos(vPolarizer1,*vUFit))
vPolDeg += [polDeg(Nmin,Nmax)]

vFalseCoin += [falseCoin(vDet0,vDet1)]

print(vFalseCoin)
pl.printAsLatexTable(np.array([['${:.2ufL}$'.format(x) for i,x in enumerate(vPolDeg)]]),
					 colTitles=["Rate %i" % i for i in range(1,6)],
					 mathMode=False)

# # Create a legend for the first line.
# first_legend = plt.legend(handles=[det0line], loc=3, bbox_to_anchor=(1,1), ncol=2, borderaxespad=0)
#
# # Add the legend manually to the current Axes.
# ax[2,0].add_artist(first_legend)
#
# # Create another legend for the second line.
# plt.legend(handles=[coin01line], loc=3, bbox_to_anchor=(1,1), ncol=2, borderaxespad=0)


fig.savefig("Figures/" + "Nonentagled_curves")



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