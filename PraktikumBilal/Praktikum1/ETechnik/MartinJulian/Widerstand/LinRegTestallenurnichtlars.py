# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:33:13 2016

@author: Julian
"""
import numpy as np
import matplotlib.pyplot as plt
import Praktikum as p

I = np.zeros(16)
I_min = np.zeros(16)
I_max = np.zeros(16)
U = np.zeros(16)
U_min = np.zeros(16)
U_max = np.zeros(16)

I[0]=0.0022
I[1]=0.0023
I[2]=0.0022
I[3]=0.0023
I[4]=0.0042
I[5]=0.0042
I[6]=0.0042
I[7]=0.0043
I[8]=0.0062
I[9]=0.0062
I[10]=0.0062
I[11]=0.0062
I[12]=0.0080
I[13]=0.0081
I[14]=0.0080
I[15]=0.0080

U[0]=1.95
U[1]=1.95
U[2]=1.95
U[3]=1.95
U[4]=3.88
U[5]=3.88
U[6]=3.88
U[7]=3.88
U[8]=5.81
U[9]=5.82
U[10]=5.82
U[11]=5.82
U[12]=7.66
U[13]=7.69
U[14]=7.68
U[15]=7.64

i=0
while i<=15:
    if(i<=len(I)):
        I_max[i]+=(0.02*I[i]+0.005*0.1)+I[i]
        i+=1
        

i=0
while i<=15:
    if(i<=len(U)):
        U_max[i]+=(0.01*U[i]+0.05)+U[i]
        i+=1

i=0
while i<=15:
    if(i<=len(I)):
        I_min[i]+=-(0.02*I[i]+0.005*0.1)+I[i]
        i+=1
        

i=0
while i<=15:
    if(i<=len(U)):
        U_min[i]+=-(0.01*U[i]+0.05)+U[i]
        i+=1



print I_max
print I_min
print U_max
print U_min
'''
fig1,ax1=plt.subplots()
ax1.scatter(I,U,marker=".")
ax1.set_xlabel('I')
ax1.set_ylabel('U')
ax1.set_title("R")
fig1.show()
'''
#%%
'Fehler'
#aus Rauschmessung
#Spannungsfehler in V
sig_U_stat=np.array([0.00009,0.00009,0.00009,0.00009,0.00008,0.00008,0.00008,0.00008,0.00007,0.00007,0.00007,0.00007,0.00009,0.00009,0.00009,0.00009])
sig_I_stat=np.array([0.00002,0.00002,0.00002,0.00002,0.00002,0.00002,0.00002,0.00002,0.00003,0.00003,0.00003,0.00003,0.000026,0.000026,0.000026,0.000026])

'Lineare Regression'
lin_reg=p.lineare_regression_xy(I,U,sig_I_stat,sig_U_stat)
lin_reg_plus=p.lineare_regression_xy(I_min,U_max,sig_I_stat,sig_U_stat)
lin_reg_min=p.lineare_regression_xy(I_max,U_min,sig_I_stat,sig_U_stat)
print lin_reg
print lin_reg_plus
print lin_reg_min
x=np.linspace(0,0.01,num=1000)
plt.xlim(0,0.01)
plt.plot(x,lin_reg[0]*x+lin_reg[2], label='R=988,56*I-0,29')
plt.plot(x,lin_reg_plus[0]*x+lin_reg_plus[2], label='R_Plus=1018,83*I+0,27')
plt.plot(x,lin_reg_min[0]*x+lin_reg_min[2], label='R_Minus=959,49*I-0,81')
plt.grid()
#%%
'Errorbars'
plt.errorbar(I,U,sig_I_stat,sig_U_stat,fmt='.')
plt.errorbar(I_min,U_max,sig_I_stat,sig_U_stat,fmt='.')
plt.errorbar(I_max,U_min,sig_I_stat,sig_U_stat,fmt='.')
plt.xlabel('I in A')
plt.ylabel('U in V')
plt.title('Messung des systematischen Fehlers 1kOhm')
plt.legend(loc='2')

#%%
sigma_Rsys=np.sqrt((lin_reg_plus[1]/2)**2+(lin_reg_min[1]/2)**2)
Rsys=lin_reg_plus[0] - lin_reg_min[0]
print "sig_Rsys=" ,Rsys, "+/-" ,sigma_Rsys

print
print
'Schnittpunkte'
S_plus = (lin_reg_plus[2]-lin_reg[2])/(lin_reg[0]-lin_reg_plus[0])
S_plus_y = lin_reg[0] * S_plus + lin_reg[2]
print(S_plus, S_plus_y)
print("Schnittpunkte der plus-Anpassung mit der mittleren Linreg")

S_min = (lin_reg_min[2]-lin_reg[2])/(lin_reg[0]-lin_reg_min[0])
S_min_y = lin_reg[0] * S_min + lin_reg[2]
print(S_plus, S_min_y)
print("Schnittpunkte der minus-Anpassung mit der mittleren Linreg")

S = (lin_reg_min[2]-lin_reg_plus[2])/(lin_reg_plus[0]-lin_reg_min[0])
S_y = lin_reg_min[0] * S_plus + lin_reg_min[2]
print(S_plus, S_y)
print("Schnittpunkte der minus-Anpassung mit der plus_Anpassung")