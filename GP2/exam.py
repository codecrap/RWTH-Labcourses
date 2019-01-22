#-*- coding: utf-8 -*-
#
#@exam.py:
#@author: Olexiy Fedorets
#@date: Thu 28.09.2017


import matplotlib.pyplot as plt
import numpy as np
import time

deltaSdeltaM = np.ones(7)*7*10**(-6)
deltaSdeltaM[0] = 6*10**(-6)


print("DeltaS/DeltaM:",deltaSdeltaM)
print("mean:",np.mean(deltaSdeltaM))
print("std, meanStd (1/N):",np.std(deltaSdeltaM),np.std(deltaSdeltaM)/np.sqrt(len(deltaSdeltaM)))
print("std, meanStd (1/(N-1)):",np.std(deltaSdeltaM,ddof=1),np.std(deltaSdeltaM,ddof=1)/np.sqrt(len(deltaSdeltaM)))


x = np.random.uniform(0,10,10000)
print(x)

stime = time.time()
print(np.mean(x))
print('%6.6E'% (stime-time.time()) )

stime = time.time()
mean = 0
for i in range(1,len(x)+1):
	mean = mean + (x[i-1]-mean)/i
print(mean)
print(stime-time.time())

stime = time.time()
sum = 0
for i in x:
	sum += i
print(sum/len(x))
print(stime-time.time())

stime = time.time()
print(x.sum()/len(x))
print(stime-time.time())
