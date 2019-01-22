#-*- coding: utf-8 -*-
#
#@serieSK.py: Praktikum: Auswertung Elehre Versuch - Serienschwingkreis, beide Gruppen
#@author: Olexiy Fedorets
#@date: Tue 05.09.2017


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../PraktikumPyLib/")
import Praktikum as p

lab = ".lab"
datapath = "../Data/"
groups = ["OlexJonathan/","DanielPhilipp/Reihe/",]
files = ["SerieSK1Ohm","SerieSK10Ohm","SerieSK20Ohm",
		"R5", "R10", "R20"]
R1s = ["1","10","20"]
R2s = ["5","10","20"]

sigmaf = 40/np.sqrt(12)	#Hz, Frequenzauflösung/Schrittweite bei Messung: 20Hz, Gleichverteilung angenommen
						# man könnte an einigen stellen auch die Hälfte des Itervalls nehmen

# group 1: L 250W->1,301mH, C 4,735µF
# group 2: L 500W->4,776mH, C 4,719µF
# resultsG1,resultsG2 = [],[]		# hier kommen später die berechneten Güten rein
resultsQ, resultsSigma = (),()

# def removeNanInDataset(data):
# 	dim = range(len(data))
# 	for set in dim:
# 		for value in data[set]:
# 			if np.isnan(value):
# 				del value
# 				for i in dim:
# 					del data[i]
	# nanIndexes = np.where(np.isnan(data))
	# print(nanIndexes)
# data = data[~np.isnan(data)]
# np.delete(data,np.isnan(data),axis=0)

def sigmaQ(f0,f1,f2):
	Q = f0/np.fabs(f2-f1)
	sigmaDeltaf = np.sqrt(sigmaf**2+sigmaf**2)
	sigmaQ = Q*np.sqrt(sigmaf**2/f0**2 + sigmaDeltaf**2/(f2-f1)**2)
	return sigmaQ

# CASSY-Hersteller: systematische Fehler
sigmaUsys = lambda Ui,Uend: 0.01*Ui + 0.005*Uend
sigmaIsys = lambda Ii,Iend: 0.01*Ii + 0.005*Iend


def SpannungQ(U0,U1,U2,Uend,f,ax):													# U0:Startspannung(vom Kondensator), gemessene Spanungen U1:C,U2:L
	# Schnittpunkt U1/U2:															# U0 nicht fehlerbehaftet!
	UcrossIndex = np.nanargmin(np.fabs(U2-U1))
	fcross = f[UcrossIndex]
	neighbors = [U1[UcrossIndex],U1[UcrossIndex+1],U1[UcrossIndex-1],				# wir nehmen 6 Benachbarte Werte des Schnittpunkts
				U2[UcrossIndex],U2[UcrossIndex+1],U2[UcrossIndex-1]]
	Ucross = np.mean(neighbors)
	sigmaUcrossStat = np.std(neighbors)/np.sqrt(6)									# Fehler auf Mittelwert!: sigma/sqrt(N)
	sigmaUcrossSys = sigmaUsys((U1[UcrossIndex]+U2[UcrossIndex])/2, Uend)
	sigmaUcross = np.sqrt(sigmaUcrossSys**2 + sigmaUcrossStat**2)

	ax.axvline(fcross,color='k',label=r"Schnittpunkt: $%.2f\pm%.2fV, %.2f\pm%.2fHz$" % (Ucross,sigmaUcross,fcross,sigmaf)
					+"\n"+r"$Q = \frac{U_{cross}}{U_0} = %.2f\pm%.2f$" % (Ucross/U0,sigmaUcross/U0))
	ax.axhline(Ucross,color='k')
	print("Schnittpunkt:\t %.2f +/- %.2fHz, %.2f +/- %.2f +/- %.2fV" % (fcross,sigmaf,Ucross,sigmaUcrossStat/U0,sigmaUcrossSys/U0))
	print("\t U0 = %.2fV" % U0)
	print("Guete:\t Q = Ucross/U0 = %.2f +/- %.2f " % (Ucross/U0,sigmaUcross/U0))
	print("\n")
	return [Ucross/U0,sigmaUcross/U0]

def StromQ(I,f,f1,f2,ax):
	#Strom-Peak finden:
	fPeak = p.peak(f,I,f1,f2)
	fmax = f[np.nanargmin(np.fabs(f-fPeak))]
	fmaxIndex = f.tolist().index(fmax)
	Imax = I[fmaxIndex]
	sigmaImaxStat = np.fabs(I[fmaxIndex+1]-I[fmaxIndex-1])/np.sqrt(12)
	sigmaImaxSys = sigmaIsys(Imax,0.7)		#I-Messbereich: 0-0,7A
	sigmaImax = np.sqrt(sigmaImaxStat**2+sigmaImaxSys**2)
	#keine Verschiebemethode gemacht! (Fehler auf Imax=>Ieff=>fRight,fLeft)
	Ieff = Imax/np.sqrt(2)
	fRight = f[np.nanargmin(np.fabs(I[fmaxIndex:]-Ieff)) + fmaxIndex]
	fLeft = f[np.nanargmin(np.fabs(I[:fmaxIndex]-Ieff))]
	print("Peak:\t %.4f +/- %.4f +/- %.4fA, %.2f +/- %.2f Hz" % (Imax,sigmaImaxStat,sigmaImaxSys,fPeak,sigmaf))
	print("\t fPlus = %.2f +/- %.2fHz, fMinus = %.2f +/- %.2fHz" % (fRight,sigmaf,fLeft,sigmaf))
	print("Guete:\t Q = fPeak/deltaf = %.2f +/- %.2f" % (fPeak/(fRight-fLeft),sigmaQ(fPeak,fRight,fLeft)))
	ax.axvline(fPeak,color='k',label=r"Peak: $%.3f\pm%.3fA, %.2f\pm%.2f Hz$" % (Imax,sigmaImax,fPeak,sigmaf)
					+"\n"+r"$f_+ = %.2f\pm%.2fHz, f_- = %.2f\pm%.2fHz$" % (fRight,sigmaf,fLeft,sigmaf)
					+"\n"+r"$Q = \frac{f_{Peak}}{\Delta f} = %.2f\pm%.2f$" % (fPeak/(fRight-fLeft),sigmaQ(fPeak,fRight,fLeft)))
	ax.axvline(fRight,color='k')
	ax.axvline(fLeft,color='k')
	ax.axhline(Imax,color='k')
	ax.axhline(Ieff,color='k')
	print("\n")
	return (fPeak/(fRight-fLeft),sigmaQ(fPeak,fRight,fLeft))


def PhasenverschiebungQ(phi,f,ax):
	# f0 = p.peak(f,-1*np.fabs(phi),fL,fR)	# minimum statt maximum finden
	# f0index = f.tolist().index(f[np.nanargmin(np.fabs(f-f0))])
	# print(f0,f0index)
	# fRight = p.peakfinder_schwerpunkt(f,-1*np.fabs(phi+45))
	# fLeft = p.peakfinder_schwerpunkt(f,-1*np.fabs(phi-45))
	# # print(fRight,fLeft)
	# # print(f[:f0index],-1*np.fabs(phi[:f0index]+45))
	phi0 = np.nanmin(np.fabs(phi))
	phi0index = np.fabs(phi).tolist().index(phi0)
	f0 = f[phi0index]
	phiPlus45index = np.nanargmin(np.fabs(phi[:phi0index]-45))
	phiMinus45index = np.nanargmin(np.fabs(phi[phi0index:]+45)) + phi0index
	fRight = f[phiMinus45index]
	fLeft = f[phiPlus45index]
	print("\t f0 = %.2f +/- %.2fHz, fPlus = %.2f +/- %.2fHz, fMinus = %.2f +/- %.2fHz" % (f0,sigmaf,fRight,sigmaf,fLeft,sigmaf))
	print("Guete (aus phi):\t Q = f0/deltaf = %.2f +/- %.2f" % (f0/(fRight-fLeft),sigmaQ(f0,fRight,fLeft)))
	ax.axvline(f0,color='k',label=r"$f_{+45} = %.2f\pm%.2fHz, f_{-45} = %.2f\pm%.2fHz$" % (fLeft,sigmaf,fRight,sigmaf)
					+"\n"+r"$f_0 = %.2f\pm%.2fHz$" % (f0,sigmaf)
					+"\n"+r"$Q = \frac{f_0}{\Delta f} = %.2f\pm%.2f$" % (f0/(fRight-fLeft),sigmaQ(f0,fRight,fLeft)))
	ax.axhline(phi[phiPlus45index],color='k')
	ax.axhline(phi[phiMinus45index],color='k')
	ax.axvline(f[phiPlus45index],color='k')
	ax.axvline(f[phiMinus45index],color='k')
	print("\n")
	return (f0/(fRight-fLeft),sigmaQ(f0,fRight,fLeft))

def plot_file(filename,group,R):
	# data group 1: n,t,U1,I1,phi,UB2(Kondensator),UB3(Spule),f1
	#				7V max
	# data group 2: n,t,U1,I1,phi,UA2(Spule),UB2(Kondensator),f1
	# 				R20: 21V max, R10,R5:7V max
	data = p.lese_lab_datei(datapath + group + filename + lab)

	fig, ax = plt.subplots(2,2,sharex=False,sharey=False,figsize=(16,8))
	fig.delaxes(ax[1][1])	# wir brauchen nur 3 plots

	if group=="OlexJonathan/": #1
		n,t,U1,I1,phi,UB2,UB3,f = data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]
		print("===================================")
		print(filename)

		ax[0][0].plot(f,UB2,'g-',label=r"$U_C [V]$")
		ax[0][0].plot(f,UB3,'b-',label=r"$U_L [V]$"+"\n")
		ax[0][1].plot(f,I1,'y-',label=r"$I [A]$"+"\n")
		ax[1][0].plot(f,phi,'r-',label=r"Phasenverschiebung $\phi [deg]$")
		ax[0][0].set_title("Serienschwingkreis " + R + r"$\Omega$, " + "Gruppe 1, Spannungen an C,L")
		ax[0][1].set_title("Serienschwingkreis " + R + r"$\Omega$, " + "Gruppe 1, Strom I")
		ax[1][0].set_title("Serienschwingkreis " + R + r"$\Omega$, " + r"Gruppe 1, Phasenverschiebung $\phi$")

		#Güte:
		resultsQ += (*SpannungQ(UB2[0],UB2,UB3,7,f,ax[0][0]),)
		# resultsG1.append(StromQ(I1,f,2000,2100,ax[0][1]))
		# resultsG1.append(PhasenverschiebungQ(phi,f,ax[1][0]))

	elif group=="DanielPhilipp/Reihe/": #2
		n,t,U1,I1,phi,UA2,UB2,_,f = data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8]
		print("===================================")
		print(filename)

		ax[0][0].plot(f,UB2,'g-',label=r"$U_C [V]$")
		ax[0][0].plot(f,UA2,'b-',label=r"$U_L [V]$"+"\n")
		ax[0][1].plot(f,I1,'y-',label=r"$I [A]$"+"\n")
		ax[1][0].plot(f,phi,'r-',label=r"Phasenverschiebung $\phi [deg]$")
		ax[0][0].set_title("Serienschwingkreis " + R + r"$\Omega$, " + "Gruppe 2, Spannungen an C,L")
		ax[0][1].set_title("Serienschwingkreis " + R + r"$\Omega$, " + "Gruppe 2, Strom I")
		ax[1][0].set_title("Serienschwingkreis " + R + r"$\Omega$, " + "Gruppe 2, Phasenverschiebung $\phi$")

		#Güte:
	# 	# if R==20: resultsG2.append(SpannungQ(UB2[0],UB2,UA2,21,f,ax[0][0]))	#Umax bei R=20 war 21V statt 7V
	# 	# else: resultsG2.append(SpannungQ(UB2[0],UB2,UA2,7,f,ax[0][0]))
	# 	# resultsG2.append(StromQ(I1,f,900,1200,ax[0][1]))
	# 	# resultsG2.append(PhasenverschiebungQ(phi,f,ax[1][0]))
	#
	# ax[0][0].set_xlabel(r"$f [Hz]$")
	# ax[0][0].set_ylabel(r"$U [V]$")
	# ax[0][0].legend(loc="lower right",fontsize=8)
	# ax[0][0].grid(True)
	#
	# ax[0][1].set_xlabel(r"$f [Hz]$")
	# ax[0][1].set_ylabel(r"$I [A]$")
	# ax[0][1].legend(loc="lower right",fontsize=8)
	# ax[0][1].grid(True)

	ax[1][0].set_xlabel(r"$f [Hz]$")
	ax[1][0].set_ylabel(r"$\phi [deg]$")
	ax[1][0].legend(loc="upper right",fontsize=8)
	ax[1][0].grid(True)

	fig.tight_layout()
	fig.savefig("../Plots/"+filename+".png",dpi=256)


print("Gruppe 1 (Jonathan,Olex)")
for f,R in zip(files[:3],R1s):
	plot_file(f,groups[0],R)
print("\nGruppe 2 (Daniel,Philipp)")
for f,R in zip(files[3:],R2s):
	plot_file(f,groups[1],R)


# QsG1,sigmaQsG1 = map(list,zip(*resultsG1))											# so macht man aus einer liste von tuplen zwei listen von jeweils allen tuple[0] und tuple[1]
# QsG2,sigmaQsG2 = map(list,zip(*resultsG2))
#
# print("\nGewichteter Mittelwert: (Q,sigmaQ)")
# print "G1,R=1:\t\t",p.gewichtetes_mittel(QsG1[0:2],sigmaQsG1[0:2])
# print "G1,R=10:\t",p.gewichtetes_mittel(QsG1[3:5],sigmaQsG1[3:5])
# print "G1,R=20:\t",p.gewichtetes_mittel(QsG1[6:8],sigmaQsG1[6:8])
# print "G2,R=5:\t\t",p.gewichtetes_mittel(QsG2[0:2],sigmaQsG2[0:2])
# print "G2,R=10:\t",p.gewichtetes_mittel(QsG2[3:5],sigmaQsG2[3:5])
# print "G2,R=20:\t",p.gewichtetes_mittel(QsG2[6:8],sigmaQsG2[6:8])


plt.show()

# annotate:
# ax[0][0].annotate("Guete: Q = U_0/U_cross = %.2f" % (Ucross/U0), xy=(0.65,0.5), xycoords='axes fraction')
# plt.xticks(list(plt.xticks()[0]) + extraticks)
# lines = plt.plot(x,y)
# ax = lines[0].axes
# ax.set_xticks(list(ax.get_xticks()) + extraticks)

# plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.1)         # just to make better use of space
# plt.get_current_fig_manager().window.showMaximized()
