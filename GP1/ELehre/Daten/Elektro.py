import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import io
import random

def readLabFile(file):
	'''
	CASSY LAB Datei einlesen (Version fuer python3).

	Messdaten werden anhand von Tabulatoren identifiziert.

	Gibt ein numpy-Array zurueck.

	'''
	f = open(file)
	dataSectionStarted = False
	dataSectionEnded = False
	data = ''
	for line in f:
		if '\t' in line and not dataSectionEnded:
			data += line
			dataSectionStarted = True
		if not '\t' in line and dataSectionStarted:
			dataSectionEnded = True
	f.close()
	dnew = data.encode('utf-8')
	return np.genfromtxt(io.BytesIO(dnew), unpack=True, dtype=float)

al =["1kOhm_1.lab", "1Ohm_1.lab", "5Ohm_1.lab", "10Ohm_1.lab", "15Ohm_1.lab", "43Ohm_1.lab"]
for i in al, s in range(2):
    f = readLabFile(i)     
    fs = np.array([x[3]for x in f])
    print(fs)
    
bl =["1Ohm_2.lab", "5Ohm_2.lab", "10Ohm_2.lab", "15Ohm_2.lab", "43Ohm_2.lab"]
for a in bl:
    g = readLabFile(a)
    g1 = np.array([x[3]for x in g])
    print(g1)
    
cl =["1Ohm_3.lab", "5Ohm_3.lab", "10Ohm_3.lab", "15Ohm_3.lab"]
for v in cl:
    h = readLabFile(v)
    h1 = np.array([x[3]for x in h])
    print(h1)

def fourier_fft(t,y):
	'''

	Schnelle Fourier-Transformation.

	Parameters
	----------
	t : array_like
		Zeitwerte der Datenpunkte
	y : array_like
		y-Werte der Datenpunkte

	Gibt das Fourierspektrum in Form zweier Listen (freq,amp)
	zurueck, die die Fourieramplituden als Funktion der zugehoerigen
	Frequenzen enthalten.
	'''
	dt = (t[-1]-t[0])/(len(t)-1)
	amp = abs(scipy.fftpack.fft(y))
	freq = scipy.fftpack.fftfreq(t.size,dt)
	return (freq,amp)  

#fourier_fft(,y)

