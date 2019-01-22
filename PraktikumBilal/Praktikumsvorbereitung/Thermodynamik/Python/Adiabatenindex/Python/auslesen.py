# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import einlesen  
import os
import matplotlib.pyplot as plt
os.chdir("C:\Users\lars\Desktop\UNI\Python\Praktikumsvorbereitung\Teil1-2016\Adiabatenindex")
os.chdir("Messung")
directory=os.listdir(os.getcwd())
print directory
for k in xrange(len(directory)):
    os.chdir(directory[k])
    hoehen = os.listdir(os.getcwd())
    #print hoehen
    for j in xrange(len(hoehen)):
        os.chdir(hoehen[j])
        einzelmessungen  = os.listdir(os.getcwd())
        #print einzelmessungen
        for i in xrange(len(einzelmessungen)):
            p = einlesen.PraktLib(einzelmessungen[i],"cassy")
            #print os.getcwd()
            #
            #
            #
            ####
            data = p.getdata()
            messpunkt = data[:,0]
            zeit = data[:,1]
            druck = data[:,2]
            ####
            plt.figure()
            plt.plot(zeit,druck)
            plt.title(directory[k]+' '+hoehen[j]+' '+einzelmessungen[i])
            plt.xlabel('t in ms')
            plt.ylabel('p in hPa')
            plt.show()
        os.chdir('..')
    os.chdir('..')
#print data
#%%
