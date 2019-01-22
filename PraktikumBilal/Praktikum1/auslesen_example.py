# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:28:59 2016

@author: lars
"""
import os
import einlesen
import matplotlib.pyplot as plt

os.chdir("C:\Users\lars\Desktop\UNI\Python\Praktikumsvorbereitung\Teil1-2016\Adiabatenindex\Messung\Druckmessung_Mittel")
os.chdir("30cm")
directory=os.listdir(os.getcwd())
print directory
for csv in directory:
    p = einlesen.PraktLib(csv,'cassy')
    data = p.getdata()
    messpunkt = data[:,0]
    zeit = data[:,1]
    druck = data[:,2]
    ####
    plt.figure()
    plt.plot(zeit,druck)
    plt.title(csv)
    plt.xlabel('t in ms')
    plt.ylabel('p in hPa')
    plt.show()