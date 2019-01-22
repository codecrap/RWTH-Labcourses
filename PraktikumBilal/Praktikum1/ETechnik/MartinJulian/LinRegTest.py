# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:33:13 2016

@author: Julian
"""
import numpy
import matplotlib.pyplot as plt

I = numpy.zeros(16)
I_min = numpy.zeros(16)
I_max = numpy.zeros(16)
U = numpy.zeros(16)
U_min = numpy.zeros(16)
U_max = numpy.zeros(16)

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

fig1,ax1=plt.subplots()
ax1.scatter(I,U,marker=".")
ax1.set_xlabel('I')
ax1.set_ylabel('U')
ax1.set_title("R")
fig1.show()