#! /usr/bin/env python

import Praktikum
import numpy as np
from pylab import *

data = Praktikum.lese_lab_datei('lab/Pendel.lab')
timeValues = data[:,1]
voltage = data[:,2]
voltageError = 0. * voltage + 0.01
offset = Praktikum.gewichtetes_mittel(voltage, voltageError)[0]
voltage = voltage - offset

figure(1)
title('Pendel')

subplot(2,1,1)
plot(timeValues, voltage)
grid()
xlabel('Zeit / s')
ylabel('Spannung / V')
einhuellende = Praktikum.exp_einhuellende(timeValues, voltage, voltageError)
plot(timeValues, +einhuellende[0] * exp(-einhuellende[2] * timeValues))
plot(timeValues, -einhuellende[0] * exp(-einhuellende[2] * timeValues))

subplot(2,1,2)
fourier = Praktikum.fourier_fft(timeValues, voltage)
frequency = fourier[0]
amplitude = fourier[1]
plot(frequency, amplitude)
grid()
xlabel('Frequenz / Hz')
ylabel('Amplitude')

maximumIndex = amplitude.argmax();
xlim(frequency[max(0, maximumIndex-10)], frequency[min(maximumIndex+10, len(frequency))])
peak = Praktikum.peakfinder_schwerpunkt(frequency, amplitude)
axvline(peak)

L = 0.666
g = ((2 * np.pi * peak)**2) * L

print 'g = %f m/s^2' % g

show()
