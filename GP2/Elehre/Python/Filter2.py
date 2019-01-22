# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 17:46:03 2017

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import Praktikum as p 

'''
Olex
Jonathan
'''

#Theorie
R47=46.69
C=4.735*10**(-6)
#Unsicherheiten
sR47=np.max([0.0025*R47,0.01/np.sqrt(12)])
sC=np.max([0.0025*C,0.001*10**(-6)/np.sqrt(12)])
print('c',C,'sc',sC)
#Grenzfrequenz: Hier ist f= (2)**(-0.5)
f_0_47=1/(R47*C*2*np.pi)
#Unsicherheit
sf_0_47=np.sqrt((1/(R47*2*np.pi*C**2))**2*sC**2+(1/(C*2*np.pi*R47**2))**2*sR47**2)
print('Grenzfrequenzen Theorie')
print('47 Ohm:',f_0_47,'+/-',sf_0_47)
#Übertragungsfunktion
def f_Hochpass(freq,R,C):
    function=1/(np.sqrt(((1/(freq*C*R*2*np.pi))**2)+1))
    return(function)

def f_Tiefpass(freq,R,C):
    function=1/(np.sqrt(((freq*C*R*2*np.pi)**2)+1))
    return(function)

"""
Daten einlesen
"""
data47=p.lese_lab_datei('C:\Users\Julia\Desktop\Elehre\Data\Data\OlexJonathan\HPTP.lab')
"""
Daten extrahieren
"""
#47 Ohm
BE_U=7
freq1=data47[1:,7]
UB21=data47[1:,5]       #Wiederstand
UA21=data47[1:,2]       #Eingangsspannung
UB31=data47[1:,6]       #Kondensator
f_31=UB31/UA21          #Tiefpass
f_21=UB21/UA21          #Hochpass

#Unsicherheiten
sUB21=0.01*UB21+0.005*BE_U
sUA21=0.01*UA21+0.005*BE_U
sUB31=0.01*UB31+0.005*BE_U
sf_31=np.sqrt((UA21)**(-2)*(sUB31)**2+(UB31/UA21**2)**2*(sUA21)**2)
sf_21=np.sqrt((UA21)**(-2)*(sUB21)**2+(UB21/UA21**2)**2*(sUA21)**2)

#print(freq,'sd')
"""
plotte Übertragungsfunktionen gegen Frequenz für 47 Ohm
"""
fig1, (ax470, ax471) = plt.subplots(nrows=2, sharex=True)
ax470.errorbar(freq1,f_21,sf_21,capsize=1)
ax470.set_xlabel('Hz')
ax470.set_ylabel('U_a/U_e')
ax471.set_xlabel('Hz')
ax471.set_ylabel('U_a/U_e')
#Theorie
theo_47_Hoch=f_Hochpass(freq1,R47,C)
ax470.plot(freq1,theo_47_Hoch,'.r')
#ax470.grid('on')

ax470.set_title('Uebertragungsfunktion_Hochpass_47')
ax471.errorbar(freq1,f_31,sf_31,capsize=1)
#Theorie
theo_47_Tief=f_Tiefpass(freq1,R47,C)
ax471.plot(freq1,theo_47_Tief,'.r')
#ax471.grid('on')
ax471.set_title('Uebertragungsfunktion_Tiefpass_47')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig1.tight_layout()        
'''
Bestimme anhand der Messung die Grenzfrequenz.
Bestimme dafür das minimum von f-2**(-0.5)
'''

#Index Bestimmung
index_f_grenz_47_Hoch=np.argmin(abs(f_21-(2**(-0.5))))
index_f_grenz_47_Tief=np.argmin(abs(f_31-2**(-0.5)))
#Werte
f_grenz_47_Hoch=freq1[index_f_grenz_47_Hoch]
f_grenz_47_Tief=freq1[index_f_grenz_47_Tief]
#Unsicherheit Gleichverteilung in +/- 20 Intervall
sGrenz0=20/(np.sqrt(3))
sGrenz_Hoch_47=np.sqrt(sGrenz0**2+sf_21[index_f_grenz_47_Hoch]**2)
sGrenz_Tief_47=np.sqrt(sGrenz0**2+sf_31[index_f_grenz_47_Tief]**2)
print('Durch Messung bestimmte Grenzfrequenzen:')
print('Hoch(47 Ohm):',f_grenz_47_Hoch,'+/-',sGrenz_Hoch_47)
print('Tief(47 Ohm):',f_grenz_47_Tief,'+/-',sGrenz_Tief_47)
'''
Abweichung von der Theorie
'''
print('abweichung_Grenzfrequenz_47_Hoch',abs(f_grenz_47_Hoch-f_0_47)/np.sqrt(sGrenz_Hoch_47**2+sf_0_47**2))
print('abweichung_Grenzfrequenz_47_Tief',abs(f_grenz_47_Tief-f_0_47)/np.sqrt(sGrenz_Tief_47**2+sf_0_47**2))
