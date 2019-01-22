# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:13:11 2017

@author: Jonathan
"""


import Praktikumsroutinen_DW
import numpy as np
import Praktikum
from pylab import *
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import matplotlib.patches as mpatches


# Olex & Daniel
A_s = np.pi*(0.035/2.)**2
A_e = np.pi*(0.023/2.)**2
r  = 0.108
sigma = 5.670373*10**(-8)
v = 10**(-4)
c = 0.221
ec = 0.0066

def histograms(name, title):
    data = Praktikum.lese_lab_datei(name)
    time = data[:,1]
    T_data = data[:,3]
    U_data = data[:,2]
    U = np.mean(U_data)
    eU = np.std(U_data)
    T = np.mean(T_data)
    eT = np.std(T_data)

    T_bins = np.linspace(T - 10., T + 10., 100)
    U_bins = np.linspace(U - 0.5, U + 0.5, 100)
    fig, ax = plt.subplots(1,2,sharex=False,sharey=True,figsize=(16,8))
    ax[0].grid(True)
    ax[0].hist(T_data, T_bins)
    ax[0].set_title(title)
    ax[0].axvline(T, color='r')
    ax[0].set_xlabel("Temperatur T [K]")
    ax[0].set_ylabel("Anzahl n")
    ax[0].set_xlim([T - 1. ,T + 1.])
    red_patch = mpatches.Patch(color='red', label='T = $%.3f\pm%.3f$' %(T, eT))
    ax[0].legend(handles=[red_patch])

    ax[1].hist(U_data, U_bins)
    ax[1].set_title(title)
    ax[1].axvline(U, color = 'r')
    ax[1].set_xlabel("Spannung U[V]")
    ax[1].set_ylabel("Anzahl n")
    ax[1].grid(True)
    ax[1].set_xlim([U - 0.04, U + 0.04])
    red_patch = mpatches.Patch(color='red', label='U = $%.3f\pm%.3f$' %(U, eU))
    ax[1].legend(handles=[red_patch])
    
    return T, eT, U, eU

def data(name):
    data = Praktikum.lese_lab_datei(name)
    time = data[:,1]
    T_data = data[:,3]
    U_data = data[:,2]
    U = np.mean(U_data)
    eU = np.std(U_data)/np.sqrt(len(U_data))
    T = np.mean(T_data)
    eT = np.std(T_data)/np.sqrt(len(T_data))
    return T, eT, U, eU


#Kalibration
T_eis, eT_eis, U_eis, eU_eis = histograms("Data/JonathanPhilipp/Rauschmessungen/Eiswassermessung.lab", "Eiswasserkalibrierung Gruppe 2")
T_sied, eT_sied, U_sied, eU_sied = histograms("Data/JonathanPhilipp/Rauschmessungen/Siedetemperatur.lab", "Siedepunktkalibrierung Gruppe 2")
steigung = 100./(T_sied - T_eis)
abschnitt = 273.15 - steigung * T_eis

print("==========================================")
print("Kalibration Gruppe 2: a = %.3f;    b = %.3f" %(steigung, abschnitt))


def realT(T, eT):
    return steigung*T + abschnitt, steigung*eT


# Rauschmessung T_0
T_0, eT_0, U_0, eU_0 = histograms("Data/JonathanPhilipp/Rauschmessungen/Zimmertemparatur nach 0 Volt.lab", "Messung T_0")
T_0, eT_0 = realT(T_0, eT_0)
print("T_0 = %.3f +/- %.3f" %(T_0, eT_0))

#Auswertung
namelist = ["Schwarz", "Weiss", "Messing", "Spiegel"]


for stoff in namelist:
        
    #globale Variablen für T und U
    T = []
    eT = []
    U = []
    eU = []
    
    # Index der Temperatur
    T_k = 50
    if stoff == "Spiegel" or stoff == "Messing":
        while T_k <= 95:
            T_i, eT_i, U_i, eU_i = data("Data/JonathanPhilipp/" + stoff +"/"+str(T_k) + "Grad.lab")
            T += [T_i]
            eT += [eT_i]
            U += [U_i]
            eU += [eU_i]
            T_k += 5
    else:
        while T_k <= 75:
            T_i, eT_i, U_i, eU_i = data("Data/JonathanPhilipp/" + stoff +"/"+str(T_k) + "Grad.lab")
            T += [T_i]
            eT += [eT_i]
            U += [U_i]
            eU += [eU_i]
            T_k += 5
            
    if stoff == "Weiss" or stoff == "Schwarz":
        T_j = 70
        while T_j <= 95:
            T_i, eT_i, U_i, eU_i = data("Data/JonathanPhilipp/" + stoff +"/"+str(T_j) + "Grad_30.lab")
            T += [T_i]
            eT += [eT_i]
            U += [U_i]
            if eU_i < 0.0000001:
                eU += [0.001/np.sqrt(12.)]
            else:
                eU += [eU_i]
            T_j += 5

    T, eT = realT(np.array(T), np.array(eT))
    U = np.array(U)
    eU = np.array(eU)
    sigma_x = np.sqrt(16*T**6*eT**2 + 4*T_0**6*eT_0**2)
    a, ea, b, eb, chi = Praktikumsroutinen_DW.residuen(T**4 - T_0**4, U, sigma_x, eU, "[K^4]", "[V]", "T^4 - T_0^4 ", "Spannung ", title = stoff + " Gruppe 2",k = T[0]**4 - T_0**4 + 0.25, l=U[6], o=T[0]**4 - T_0**4 , p = -0.04)
    print (stoff + " Gruppe 2 : a = %.15f +/- %.15f;    b = %.3f +/- %.3f" %(a, ea, b, eb))
    
    
    epsilon = (a*v*r**2*np.pi)/(c*A_s*A_e*sigma)
    sigma_epsilon_stat = (v*r**2*np.pi)/(c*A_s*A_e*sigma)*ea
    sigma_epsilon_sys = (a*v*r**2*np.pi)/(c**2*A_s*A_e*sigma)*ec
    if stoff == "Schwarz":
        e_s = epsilon
        s_stat = sigma_epsilon_stat
        s_sys = sigma_epsilon_sys
    print ("     epsilon = %.4f +/- %.4f (stat) +/- %.4f (sys)" %(epsilon, sigma_epsilon_stat, sigma_epsilon_sys))
    rel = epsilon/e_s
    erel_stat = np.sqrt((epsilon/e_s**2 * sigma_epsilon_stat)**2 + (1./e_s * s_stat)**2)
    erel_sys = np.sqrt((epsilon/e_s**2 * sigma_epsilon_sys)**2 + (1./e_s * s_sys)**2)
    print ("     relativ zu schwarzer Seite: %.3f +/- %.3f +/- %.3f" %(rel, erel_stat, erel_sys))
    
     #Auskommentiert, da sonst die lineare Regression für Messing nicht geplottet wird
    if stoff == "Messing":
        print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        def f(T, p0, p1, p2):
            return p0 + p1*T**p2
    
        p_opt, p_cov = curve_fit(f, T, U, np.array([b, a, 4.])) # klappt manchmal nur ohne den -a*T_0^4 Term
        p_err = np.sqrt(np.diag(p_cov))
        print ("Fit: U = p0 + p1 * T^p2")
        print("Fitparameter: p0 = %.3f +/- %.3f" %(p_opt[0], p_err[0]))
        print("              p1 = %.15f +/- %.15f" %(p_opt[1], p_err[1]))
        print("              p2 = %.3f +/- %.3f" %(p_opt[2], p_err[2]))
        ax = plt.subplot(111)
        ax.plot(T, f(T, p_opt[0], p_opt[1], p_opt[2]), color = 'b')
        ax.errorbar(T, U, eT, eU, color = 'r', fmt = 'o', markersize = 5)
        ax.set_title("T^4 Fit an Messwerte fuer " + str(stoff))
        ax.set_ylabel("Spannung [V]")
        ax.set_xlabel("Temperatur [K]")
        red_patch = mpatches.Patch(color='red', label='Messwerte')
        blue_patch = mpatches.Patch(color = 'blue', label = 'Fit')
        ax.legend(handles=[red_patch, blue_patch])
        ax.grid(True)
        
        
    print ("--------------------------------------------------------------------------------")
    
    
