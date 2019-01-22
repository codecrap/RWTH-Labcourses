# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 19:37:12 2016

@author: Erik
"""

import Praktikum
import numpy as np
import einlesen



pfad = "Kuhlung_1.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_1   =  (druck[0:1*len(druck)/16+1]).mean()
druck_2   =  (druck[1*len(druck)/16+1:2*len(druck)/16+1]).mean()
druck_3   =  (druck[2*len(druck)/16+1:3*len(druck)/16+1]).mean()
druck_4   =  (druck[3*len(druck)/16+1:4*len(druck)/16+1]).mean()
druck_5   =  (druck[4*len(druck)/16+1:5*len(druck)/16+1]).mean()
druck_6   =  (druck[5*len(druck)/16+1:6*len(druck)/16+1]).mean()
druck_7   =  (druck[6*len(druck)/16+1:7*len(druck)/16+1]).mean()
druck_8   =  (druck[7*len(druck)/16+1:8*len(druck)/16+1]).mean()
druck_9   =  (druck[8*len(druck)/16+1:9*len(druck)/16+1]).mean()
druck_10  =  (druck[9*len(druck)/16+1:10*len(druck)/16+1]).mean()
druck_11  =  (druck[10*len(druck)/16+1:11*len(druck)/16+1]).mean()
druck_12  =  (druck[11*len(druck)/16+1:12*len(druck)/16+1]).mean()
druck_13  =  (druck[12*len(druck)/16+1:13*len(druck)/16+1]).mean()
druck_14  =  (druck[13*len(druck)/16+1:14*len(druck)/16+1]).mean()
druck_15  =  (druck[14*len(druck)/16+1:15*len(druck)/16+1]).mean()
druck_16  =  (druck[15*len(druck)/16+1:16*len(druck)/16+1]).mean()
P_G1_1=np.array([druck_1,druck_2,druck_3,druck_4,druck_5,druck_6,druck_7,druck_8,druck_9,druck_10,druck_11,druck_12,druck_13,druck_14,druck_15,druck_16])

#%%

pfad = "Kuhlung_2.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_1   =  (druck[0:1*len(druck)/16+1]).mean()
druck_2   =  (druck[1*len(druck)/16+1:2*len(druck)/16+1]).mean()
druck_3   =  (druck[2*len(druck)/16+1:3*len(druck)/16+1]).mean()
druck_4   =  (druck[3*len(druck)/16+1:4*len(druck)/16+1]).mean()
druck_5   =  (druck[4*len(druck)/16+1:5*len(druck)/16+1]).mean()

P_G1_2=np.array([druck_1,druck_2,druck_3,druck_4,druck_5])

P_G1=np.concatenate((P_G1_1,P_G1_2))


#%%


pfad = "Hauptmessung.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_3   =  (druck[2*len(druck)/16+1:3*len(druck)/16+1]).mean()
druck_4   =  (druck[3*len(druck)/16+1:4*len(druck)/16+1]).mean()
druck_5   =  (druck[4*len(druck)/16+1:5*len(druck)/16+1]).mean()
druck_6   =  (druck[5*len(druck)/16+1:6*len(druck)/16+1]).mean()
druck_7   =  (druck[6*len(druck)/16+1:7*len(druck)/16+1]).mean()
druck_8   =  (druck[7*len(druck)/16+1:8*len(druck)/16+1]).mean()
druck_9   =  (druck[8*len(druck)/16+1:9*len(druck)/16+1]).mean()
druck_10  =  (druck[9*len(druck)/16+1:10*len(druck)/16+1]).mean()
druck_11  =  (druck[10*len(druck)/16+1:11*len(druck)/16+1]).mean()
druck_12  =  (druck[11*len(druck)/16+1:12*len(druck)/16+1]).mean()
P_G2=np.array([druck_3,druck_4,druck_5,druck_6,druck_7,druck_8,druck_9,druck_10,druck_11,druck_12])

#%%

print P_G1,"\n"
print P_G2