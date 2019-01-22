# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import Praktikum as p 

'''
Phillip
Daniel
'''

#Theorie
R47=46.55
R20=19.83
C=4.719*10**(-6)
#Unsicherheiten
sR47=np.max([0.0025*R47,0.01/np.sqrt(12)])
sR20=np.max([0.0025*R20,0.01/np.sqrt(12)])
sC=np.max([0.000025*C,0.001*10**(-6)/np.sqrt(12)])
#Grenzfrequenz: Hier ist f= (2)**(-0.5)
f_0_47=1/(R47*C*2*np.pi)
f_0_20=1/(R20*C*2*np.pi)
#Unsicherheit
sf_0_47=np.sqrt((1/(R47*2*np.pi*C**2))**2*sC**2+(1/(C*2*np.pi*R47**2))**2*sR47**2)
sf_0_20=np.sqrt((1/(R20*2*np.pi*C**2))**2*sC**2+(1/(C*2*np.pi*R20**2))**2*sR20**2)
print('Grenzfrequenzen Theorie')
print('47 Ohm:',f_0_47,'+/-',sf_0_47)
print('20 Ohm:',f_0_20,'+/-',sf_0_20)
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
data20=p.lese_lab_datei('C:\Users\Julia\Desktop\Elehre\Data\Data\DanielPhilipp\Filter\Hoch und Tief R20C4.7mu.lab')
data47=p.lese_lab_datei('C:\Users\Julia\Desktop\Elehre\Data\Data\DanielPhilipp\Filter\Hoch und Tief R47C4.7mu.lab')
#data2=p.lese_lab_datei('Schwebung Abstand=0/Schwebung3.lab')

"""
Daten extrahieren
"""
#47 Ohm
BE_U=7
freq1=data47[1:,9]
f_31=data47[1:,11]      #Tiefpass
f_21=data47[1:,10]      #Hochpass
UB21=data47[1:,6]
UA21=data47[1:,5]
UB31=data47[1:,8]
#Unsicherheiten
sUB21=0.01*UB21+0.005*BE_U
sUA21=0.01*UA21+0.005*BE_U
sUB31=0.01*UB31+0.005*BE_U
sf_31=np.sqrt((UA21)**(-2)*(sUB31)**2+(UB31/UA21**2)**2*(sUA21)**2)
sf_21=np.sqrt((UA21)**(-2)*(sUB21)**2+(UB21/UA21**2)**2*(sUA21)**2)
#20 Ohm

freq=data20[1:,9]        #Frequenzverlauf
f_3=data20[1:,11]        #Übertragungsfunktion Tiefpass =ub3/ua2
f_2=data20[1:,10]        #Übertragungsfunktion Hochpass =ub2/ua2
UB2=data20[1:,6]
UA2=data20[1:,5]
UB3=data20[1:,8]
#Unsicherheiten
sUB2=0.01*UB2+0.005*BE_U
sUA2=0.01*UA2+0.005*BE_U
sUB3=0.01*UB3+0.005*BE_U
sf_3=np.sqrt((UA2)**(-2)*(sUB3)**2+(UB3/UA2**2)**2*(sUA2)**2)
sf_2=np.sqrt((UA2)**(-2)*(sUB2)**2+(UB2/UA2**2)**2*(sUA2)**2)
#print(freq,'sd')
"""
plotte Übertragungsfunktionen gegen Frequenz für 47 Ohm
"""
fig47, (ax470, ax471) = plt.subplots(nrows=2, sharex=True)
ax470.errorbar(freq1,f_21,sf_21,capsize=1)
#Theorie
theo_47_Hoch=f_Hochpass(freq1,R47,C)
ax470.plot(freq1,theo_47_Hoch,'.r',linewidth=0.5)
#ax470.grid('on')

ax470.set_title('Uebertragungsfunktion_Hochpass_47')
ax471.errorbar(freq1,f_31,sf_31,capsize=1)
#Theorie
theo_47_Tief=f_Tiefpass(freq1,R47,C)
ax471.plot(freq1,theo_47_Tief,'.r',linewidth=0.5)
#ax471.grid('on')
ax471.set_title('Uebertragungsfunktion_Tiefpass_47')
ax470.set_xlabel('Hz')
ax470.set_ylabel('U_a/U_e')
ax471.set_xlabel('Hz')
ax471.set_ylabel('U_a/U_e')
fig47.tight_layout()
"""
plotte Übertragungsfunktionen gegen Frequenz für 20 Ohm
"""
fig20, (ax20, ax21) = plt.subplots(nrows=2, sharex=True)
'''Hoch'''
ax20.errorbar(freq,f_2,sf_2,capsize=1)
#Theorie
theo_20_Hoch=f_Hochpass(freq,R20,C)
ax20.plot(freq,theo_20_Hoch,'.r',linewidth=0.5)
#ax20.grid('on')
ax20.set_title('Uebertragungsfunktion_Hochpass_20')
'''Tief'''
ax21.errorbar(freq,f_3,sf_3,capsize=1)
#Theorie
theo_20_Tief=f_Tiefpass(freq,R20,C)
ax21.plot(freq1,theo_20_Tief,'.r',linewidth=0.5)
#ax21.grid('on')
ax21.set_title('Uebertragungsfunktion_Tiefpass_20')
ax20.set_xlabel('Hz')
ax20.set_ylabel('U_a/U_e')
ax21.set_xlabel('Hz')
ax21.set_ylabel('U_a/U_e')
fig20.tight_layout()
'''
Bestimme anhand der Messung die Grenzfrequenz.
Bestimme dafür das minimum von f-2**(-0.5)
'''

#Index Bestimmung
index_f_grenz_47_Hoch=np.argmin(abs(f_21-(2**(-0.5))))
index_f_grenz_47_Tief=np.argmin(abs(f_31-2**(-0.5)))
index_f_grenz_20_Hoch=np.argmin(abs(f_2-2**(-0.5)))
index_f_grenz_20_Tief=np.argmin(abs(theo_20_Tief-2**(-0.5)))
#Werte
f_grenz_47_Hoch=freq1[index_f_grenz_47_Hoch]
f_grenz_47_Tief=freq1[index_f_grenz_47_Tief]
f_grenz_20_Hoch=freq[index_f_grenz_20_Hoch]
f_grenz_20_Tief=freq[index_f_grenz_20_Tief]
#Unsicherheit Gleichverteilung in +/- 20 Intervall
sGrenz0=20/(np.sqrt(3))
sGrenz_Hoch_47=np.sqrt(sGrenz0**2+sf_21[index_f_grenz_47_Hoch]**2)
sGrenz_Tief_47=np.sqrt(sGrenz0**2+sf_31[index_f_grenz_47_Tief]**2)
sGrenz_Hoch_20=np.sqrt(sGrenz0**2+sf_2[index_f_grenz_20_Hoch]**2)
sGrenz_Tief_20=np.sqrt(sGrenz0**2+sf_3[index_f_grenz_20_Tief]**2)
print('Durch Messung bestimmte Grenzfrequenzen:')
print('Hoch(47 Ohm):',f_grenz_47_Hoch,'+/-',sGrenz_Hoch_47)
print('Hoch(20 Ohm):',f_grenz_20_Hoch,'+/-',sGrenz_Hoch_20)
print('Tief(47 Ohm):',f_grenz_47_Tief,'+/-',sGrenz_Tief_47)
print('Tief(20 Ohm):',f_grenz_20_Tief,'+/-',sGrenz_Tief_20)
'''
Abweichung von der Theorie
'''
print('abweichung_Grenzfrequenz_47_Hoch',abs(f_grenz_47_Hoch-f_0_47)/np.sqrt(sGrenz_Hoch_47**2+sf_0_47**2))
print('abweichung_Grenzfrequenz_47_Tief',abs(f_grenz_47_Tief-f_0_47)/np.sqrt(sGrenz_Tief_47**2+sf_0_47**2))
print('abweichung_Grenzfrequenz_20_Hoch',abs(f_grenz_20_Hoch-f_0_20)/np.sqrt(sGrenz_Hoch_20**2+sf_0_20**2))
print('abweichung_Grenzfrequenz_20_Tief',abs(f_grenz_20_Tief-f_0_20)/np.sqrt(sGrenz_Tief_20**2+sf_0_20**2))


