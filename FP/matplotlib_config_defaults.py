#-*- coding: utf-8 -*-
#
#@matplotlib_config.py: set default settings for plots
#@author: Olexiy Fedorets
#@date: Fri 22.02.2019


import matplotlib.pyplot as plt
import matplotlib as mpl

def set_mpl_defaults():
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

if __name__ == '__main__':
	set_mpl_defaults()
