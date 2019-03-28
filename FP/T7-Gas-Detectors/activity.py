# -*- coding: utf-8 -*-
#
# @activity.py: compute activity of radioactive sources after a given time
# @author: Olexiy Fedorets
# @date: Tue 19.02.2019


import numpy as np
import datetime as dt
import sys
import PraktLib as pl
import uncertainties.unumpy as unp
sys.path.append("./../../")

# take care of changes in module by manually reloading
from importlib import reload
pl = reload(pl)


# define start values
vNames = ["Cs (strong)","Cs (weak)","Co","Eu","Na"]
T_experiment = dt.datetime(2019, 2, 19, 12, 35, 17, 420514)
vT_start = [dt.datetime(2010,11,23),dt.datetime(1988,8,12),dt.datetime(2003,4,15),dt.datetime(1978,6,2),dt.datetime(2005,1,12)]
vActivity_start = np.array([44400,37,37,37,37]) * 10**3 							# convert kBq -> Bq
vT_halflife = unp.uarray( [11000,11000,1925.3,4943,950.5], [90,90,0.4,5,0.4] ) 		# in days

# compute activities at t=T_experiment
vDeltaT = np.array([ (T_experiment - t).total_seconds() for _,t in enumerate(vT_start) ])
vLambda =  np.log(2)/(vT_halflife *24*60*60)  										# convert days -> seconds
# print(vT_halflife,vLambda)
fActivity = lambda A0,l,t: A0 * unp.exp(-l*t)
vActivity_today = fActivity(vActivity_start,vLambda,vDeltaT)


print(vActivity_today)
pl.printAsLatexTable( np.array([ [x.strftime("%d.%m.%Y") for _,x in enumerate(vT_start)],
								['${:.0f}$'.format(x*10**-3) for _,x in enumerate(vActivity_start)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vT_halflife)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vLambda)],
								['${:.1ueL}$'.format(x*10**-3) for _,x in enumerate(vActivity_today)] ]),
					colTitles=vNames,
					rowTitles=["Buy date","Activity at buy date (kBq)","Halflife time (days)","Decay constant (1/s)","Activity today (kBq)"],
					mathMode=False )
