# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: Lars
"""
import numpy as np

''' Rohdaten Kondensator '''
c_Lit=1*10**(-6)

'Messbereich'
t_B=10*10**(-3) #s
I_B=10 #A
U_B=10 #V

'Trigger'
U_T= -0.32 #V steigend Aufladung
U_T=1.76 #V fallend Entladung

'Ablesefehler'
sig_T=0.04*10**(-3)*1/np.sqrt(12)
sig_I=0.04*1/np.sqrt(12)
sig_U=0.04*1/np.sqrt(12)

'''Aufladung'''
'Offset'
off_I=80*10**(-3)
'1.'
delta_t_a1=10**(-3) #s
I1_2=0.72-off_I #A
I1_1=1.88-off_I #A
'2.'
delta_t_a2=10**(-3) #s
I2_2=0.84-off_I #A
I2_1=2.24-off_I #A
'3.'
delta_t_a3=10**(-3) #s
I3_2=0.76-off_I #A
I3_1=2.12-off_I #A
'4.'
delta_t_a4=10**(-3) #s
I4_2=0.72-off_I #A
I4_1=1.92-off_I #A

'''Entladung'''
'Offset'
off_U=80*10**(-3)
'1.'
delta_t_e1=10**(-3) #s
U1_2=0.84-off_U #A
U1_1=2.48-off_U #A
'2.'
delta_t_e2=10**(-3) #s
U2_2=0.84-off_U #A
U2_1=2.44-off_U #A
'3.'
delta_t_e3=10**(-3) #s
U3_2=0.8-off_U #A
U3_1=2.4-off_U #A
'4.'
delta_t_e4=10**(-3) #s
U4_2=0.88-off_U #A
U4_1=2.44-off_U #A


R=988.565
sig_R=5.45
