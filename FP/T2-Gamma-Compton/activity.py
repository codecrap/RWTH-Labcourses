#-*- coding: utf-8 -*-
#
#@activity.py: compute activity of sample after a given time using date conversion
#@author: Olexiy Fedorets
#@date: Tue 19.02.2019


import numpy as np
import datetime as dt
import sys
sys.path.append("./../../")
import PraktLib as pl

vNames =  ["Cs-strong","Cs-weak","Co","Eu","Na"]
vT_start = [dt.datetime(2010,11,23),dt.datetime(1988,8,12),dt.datetime(2003,4,15),dt.datetime(1978,6,2),dt.datetime(2005,1,12)]
vDeltaT = np.array([ (dt.datetime.today() - vT_start[i]).total_seconds() for i,_ in enumerate(vT_start) ])
vActivity_start = np.array([44400,37,37,37,37]) * 10**3

vT_halflife = np.array([11000,11000,1925.3,4943,950.5]) *24*60*60
vErrors_Thalflife = np.array([90,90,0.4,5,0.4]) *24*60*60
vLambda =  np.log(2)/vT_halflife

Factivity = lambda A0,l,t: A0 * np.exp(-l*t)
vActivity_today = Factivity(vActivity_start,vLambda,vDeltaT)


print(vActivity_today)

pl.printAsLatexTable(np.array([ [vT_start[i].strftime("%d.%m.%Y") for i,_ in enumerate(vT_start)],
								vActivity_start,vT_halflife,vErrors_Thalflife,vLambda,vActivity_today]),
								colTitles=vNames,
								rowTitles=["Buy date","Activity at buy date","Halflife time","Halflife time errors","Decay constant","Activity today"],
								decimals=1,
								mathMode=False)
