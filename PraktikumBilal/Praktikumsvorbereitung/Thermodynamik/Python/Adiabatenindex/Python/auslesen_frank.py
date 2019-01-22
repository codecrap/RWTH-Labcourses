# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 12:09:56 2016

@author: lars
"""

import einlesen  
import os
import matplotlib.pyplot as plt
os.chdir("C:\Users\lars\Desktop\UNI\Python\Praktikumsvorbereitung\Teil1-2016\Adiabatenindex")
os.chdir("Frank_Pierre")
directory=os.listdir(os.getcwd())
for i in directory:
    #print i
    p = einlesen.PraktLib(i,'cassy')
    data = p.getdata()
    messpunkt = data[:,0]
    zeit = data[:,1]
    druck = data[:,2]
    ####
    plt.figure()
    plt.plot(zeit,druck)
    i=i.replace('\xf6','oe')
    plt.title(i)
    plt.xlabel('t in ms')
    plt.ylabel('p in hPa')
    plt.show()