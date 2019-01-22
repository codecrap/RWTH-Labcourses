#-*- coding: utf-8 -*-
#
#@auswertung_thermo.py:
#@author: Olexiy Fedorets
#@date: Sun 03.12.2017


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import PraktLib as pl
import operator

fontsize = 20

n,t,p = pl.readLabFile("./Data/rauschen.lab")
p0 = np.mean(p)
sigmap0 = np.std(p)/np.sqrt(len(p))
print("p0 = %.3f \pm %e" % (p0,sigmap0))

for f in ["gross_hoch_2","mittel_mitte_2","klein_mitte_2"]:
	n,t,p = pl.readLabFile("./Data/"+f+".lab")
	fig,ax = plt.subplots()
	ax.plot(t,p-p0,'b-')
	ax.set_title(f,fontsize=fontsize)
	ax.set_xlabel("Zeit [s]",fontsize=fontsize)
	ax.set_ylabel("Druck [hPa]",fontsize=fontsize)
	ax.grid(True)
	fig.tight_layout()
	fig.savefig("./Plots/"+f+".pdf",format='pdf',dpi=256)



GH = [1.214,1.191,1.1738]
N_GH = [1,1,2]
GT = [1.1535,1.1588,1.169]
N_GT = [2,2,2]

KH = [0.23,0.2372,0.2435]
N_KH = [12,6,4]
KM = [0.2319,0.2209,0.2206]
N_KM = [7,6,6]
KT = [0.2246,0.2154,0.2238]
N_KT = [7,4,6]

MH = [0.4018,0.3823,0.4014]
N_MH = [4,5,3]
MM = [0.3994,0.3748,0.3968]
N_MM = [5,6,9]
MT = [0.3788,0.3964,0.3935]
N_MT = [8,5,6]

# V = [H,M,T] in cm^3 !!
V_K = np.array([378.377,362.891])*10**(-6)
V_M = np.array([1219.099,1199.018,1183.533])*10**(-6)
V_G = np.array([11400.717,11365.152])*10**(-6)
sigma_VM = np.array([0.516,0.394,0.274])*10**(-6)
sigma_VK = np.array([0.394,0.274])*10**(-6)
sigma_VG = np.array([0.547,0.330])*10**(-6)

# pl.printAsLatexTable(rowTitles=["Höhen",""])

V = np.array(list(V_G)+list(V_M)+list(V_K))
sigmaV = np.array(list(sigma_VG)+list(sigma_VM)+list(sigma_VK))
print(V,sigmaV)

def sigmaOmega(T,N):
	T = np.array(T)
	N = np.array(N)
	sigma_t = 500 * 10**(-6) #s
	omega = 2*np.pi/T
	sigma_omega = omega * (2*np.pi/T**2) * np.sqrt(2)*sigma_t/N
	return np.average(omega,weights=sigma_omega,returned=True)

omegas = []
print(KH[1:])
for Ti,Ni in zip([GH,GT,MH,MM,MT,KM,KT],[N_GH,N_GT,N_MH,N_MM,N_MT,N_KM,N_KT]):
	omegas.append(sigmaOmega(Ti,Ni))

print(omegas)


# print(dict(omegas).keys(),list(map(lambda x: x[0],omegas)),map(operator.itemgetter(0),omegas))

pl.plotFit(1/V, sigmaV/(V**2), np.array(list(map(lambda x: x[0]**2,omegas))), np.array(list(map(lambda x: x[1]*2*x[0],omegas))),
 			title="Kappa-Bestimmung",xlabel=r"$1/V \;[m^{-3}]$",ylabel=r"$\omega^2 \;[Hz^2]$",res_ylabel=r"$\omega^2 - (a \cdot 1/V + b)$")


# pl.plotFit(1/np.sqrt(V_K), sigma_VKM/(2*np.power(V_K,3./2.0)),
#  			np.array(list(map(lambda x: x[0],omegas[-3:]))), np.array(list(map(lambda x: x[1],omegas[-3:]))),
#  			title="Kappa-Bestimmung",xlabel=r"$1/\sqrt{V} \;[\sqrt{cm^{-3}}]$",ylabel=r"$\omega \;[Hz]$")
#
# pl.plotFit(1/np.sqrt(V_M), sigma_VKM/(2*V_M**(3./2.)), np.array(list(map(lambda x: x[0],omegas[2:5]))), np.array(list(map(lambda x: x[1],omegas[2:5]))),
#  			title="Kappa-Bestimmung",xlabel=r"$1/\sqrt{V} \;[\sqrt{cm^{-3}}]$",ylabel=r"$\omega \;[Hz]$")
#
# pl.plotFit(1/np.sqrt(V_G), sigma_VG/(2*V_G**(3./2.)), np.array(list(map(lambda x: x[0],omegas[0:2]))), np.array(list(map(lambda x: x[1],omegas[0:2]))),
#  			title="Kappa-Bestimmung",xlabel=r"$1/\sqrt{V} \;[\sqrt{cm^{-3}}]$",ylabel=r"$\omega \;[Hz]$")





plt.show()
