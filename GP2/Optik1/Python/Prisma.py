# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:06:16 2017

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import sys
sys.path.append("../../PraktikumPyLib/")
import Praktikumsroutinen_DW as res

def umrechnung(wert):
    wert=np.array(wert)
    new_wert=wert/60.
    return(new_wert)
def umrechung_1(wert):
    s=2*np.pi/360
    new_wert=wert*s
    return(new_wert)
epsilon=umrechung_1(60)

'''LINKS Kommata'''
du_vio_l=umrechung_1(umrechnung([2,1,3])+155)
vio_l=umrechung_1(umrechnung([21,20,20])+153)
bl1_l=umrechung_1(umrechnung([6,5,4])+152)
bl2_l=umrechung_1(umrechnung([39,37,41])+151)
bl_gruen_l=umrechung_1(umrechnung([52,51,53])+150)
gruen_l=umrechung_1(umrechnung([3,2,4])+150)
gelb_l=umrechung_1(umrechnung([34,33,31])+149)
rot_l=umrechung_1(umrechnung([40,38,38])+148)
'''RECHTS Kommata'''
du_vio_r=umrechung_1(umrechnung([32,31,33])+24)
vio_r=umrechung_1(umrechnung([13,14,14])+26)
bl1_r=umrechung_1(umrechnung([30,31,33])+27)
bl2_r=umrechung_1(umrechnung([52,52,53])+27)
bl_gruen_r=umrechung_1(umrechnung([41,41,49])+28)
gruen_r=umrechung_1(umrechnung([26,30,30])+29)
gelb_r=umrechung_1(umrechnung([1,0,2])+30)
rot_r=umrechung_1(umrechnung([51,52,50,52,51,52,50,50,53,51])+30)
'''winkel'''
delta_l=[]
error_delta_l=[]
delta_r=[]
error_delta_r=[]
error_rausch=np.std(rot_r)
#print(error_rausch,'Fehler')
name_l=[du_vio_l,vio_l,bl1_l,bl2_l,bl_gruen_l,gruen_l,gelb_l,rot_l]
name_r=[du_vio_r,vio_r,bl1_r,bl2_r,bl_gruen_r,gruen_r,gelb_r]
wavelength=np.array([404.66,435.83,467.81,479.99,508.58,546.07,576.96,643.85])*10**(-9)
for i in name_l:
    delta_l+=[np.mean(i)]
    error_delta_l+=[error_rausch/np.sqrt(3)]
for i in name_r:
    delta_r+=[np.mean(i)]
    error_delta_r+=[error_rausch/np.sqrt(3)]
delta_r+=[np.mean(rot_r)]
error_delta_r+=[error_rausch/np.sqrt(10)]
'''minimaler ablenkwinkel'''
delta=(np.array(delta_l)-np.array(delta_r))/2
error_delta=(np.sqrt(np.array(error_delta_l)**2+np.array(error_delta_r)**2))/2
#print(delta)
#print(error_delta)
'''Bestimmung n'''
n=np.sin((delta+epsilon)/2)/np.sin(epsilon/2)
error_n=np.cos((epsilon+delta)/2)/(2*np.sin(epsilon/2))*error_delta
#linearer Fit
a,ea,b,eb,chiq_ndof=res.residuen((1/wavelength)**2,n,0,error_n,'[$m^{-2}$]','','$\lambda^{-2}$','n',ca=0,k=2.5*10**12,l=1.76,o=3.5*10**12,p=0.001)
#print(a,b)


def func(param,lambd):
    func=param[0]*(1+param[1]/lambd**2+param[2]/lambd**4)
    return func

def diff(param,wavelength,n,n_error):
    return (n-func(param,wavelength))/n_error

#fit mit leastsq
opt_param=[1,0,0]
# "full_output=True" muss gesetzt sein, um Kovarianzmatrix zu bekommen !!
opt_param, cov, info, message, status = sc.leastsq(diff,opt_param,args=(wavelength,n,error_n),full_output=True)
print(opt_param,cov,info,message,status)

residue=n-func(opt_param,wavelength)
errs=np.sqrt(error_n**2)

chiq=sum((n-func(opt_param,wavelength))/error_n)**2/(len(n)-len(opt_param)) 	# Freiheitsgrad?
print("chiq/ndf = ", chiq)

# Fehler auf Fitparameter:				# siehe: -->
paramErrs = np.sqrt(np.diag(cov)*chiq)	# https://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
										# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i

fig10,(ax10,ax20)=plt.subplots(1,2)
ax10.plot(wavelength**(-2),func(opt_param,wavelength),'r-',
			label="$ a = %.1e \pm %.1e$, \n $b1 = %.1e \pm %.1e$, \n $b2 = %.1e \pm %.1e$"
			% (opt_param[0],paramErrs[0],opt_param[1],paramErrs[1],opt_param[2],paramErrs[2]))
ax10.errorbar(wavelength**(-2),n,error_n,0,'.g')
ax10.legend(loc="upper left")


ax20.errorbar(wavelength**(-2),residue,errs,0,'.',capsize=3,
				label=r"$\chi^2/ndf = %.3f$" % chiq)
ax20.plot(wavelength**(-2), 0*wavelength,'r')
ax20.legend(loc="upper right")

ax20.annotate(chiq,xy=(0.0002,0.75),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})


plt.show()




# sellmer=param[0]*(1+param[1]/lambd**2+param[2]/lambd**4
