# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Author: D.W.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
import scipy.odr
import StringIO

def nulllinie(datax):
    """ 
    Diese Funktion benötigt ein array für 
    Datax also die X-werte des zu Plottenden Graphen
    der mit der nulllinie verglichen werden soll.
    Und gibt bisher ein Array an werten zurück das als separate Funktion mitgeplottet werden kann.
    """
    nulllinie2=np.array([0 for x in range(len(datax))])
    x=[]
    for i in range(len(datax)):
        x+=[i]
    nulllinie1=x
    return(nulllinie1,nulllinie2)

def lineare_regression(x,y,ey):
    '''

    Lineare Regression.

    Parameters
    ----------
    x : array_like
        x-Werte der Datenpunkte
    y : array_like
        y-Werte der Datenpunkte
    ey : array_like
        Fehler auf die y-Werte der Datenpunkte

    Diese Funktion benoetigt als Argumente drei Listen:
    x-Werte, y-Werte sowie eine mit den Fehlern der y-Werte.
    Sie fittet eine Gerade an die Werte und gibt die
    Steigung a und y-Achsenverschiebung b mit Fehlern
    sowie das chi^2 und die Korrelation von a und b
    als Liste aus in der Reihenfolge
    [a, ea, b, eb, chiq, cov].
    '''

    s   = sum(1./ey**2)
    sx  = sum(x/ey**2)
    sy  = sum(y/ey**2)
    sxx = sum(x**2/ey**2)
    sxy = sum(x*y/ey**2)
    delta = s*sxx-sx*sx
    b   = (sxx*sy-sx*sxy)/delta
    a   = (s*sxy-sx*sy)/delta
    eb  = np.sqrt(sxx/delta)
    ea  = np.sqrt(s/delta)
    cov = -sx/delta
    corr = cov/(ea*eb)
    chiq  = sum(((y-(a*x+b))/ey)**2)

    return(a,ea,b,eb,chiq,corr)


def lineare_regression_xy(x,y,ex,ey):
    '''

    Lineare Regression mit Fehlern in x und y.

    Parameters
    ----------
    x : array_like
        x-Werte der Datenpunkte
    y : array_like
        y-Werte der Datenpunkte
    ex : array_like
        Fehler auf die x-Werte der Datenpunkte
    ey : array_like
        Fehler auf die y-Werte der Datenpunkte

    Diese Funktion benoetigt als Argumente vier Listen:
    x-Werte, y-Werte sowie jeweils eine mit den Fehlern der x-
    und y-Werte.
    Sie fittet eine Gerade an die Werte und gibt die
    Steigung a und y-Achsenverschiebung b mit Fehlern
    sowie das chi^2 und die Korrelation von a und b
    als Liste aus in der Reihenfolge
    [a, ea, b, eb, chiq, cov].

    Die Funktion verwendet den ODR-Algorithmus von scipy.
    '''
    a_ini,ea_ini,b_ini,eb_ini,chiq_ini,corr_ini = lineare_regression(x,y,ey)

    def f(B, x):
        return B[0]*x + B[1]

    model  = scipy.odr.Model(f)
    data   = scipy.odr.RealData(x, y, sx=ex, sy=ey)
    odr    = scipy.odr.ODR(data, model, beta0=[a_ini, b_ini])
    output = odr.run()
    ndof = len(x)-2
    chiq = output.res_var*ndof
    corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])

    return output.beta[0],output.sd_beta[0],output.beta[1],output.sd_beta[1],chiq,corr

def residuen(x,y,ex,ey,ux,uy,Parameterx,Parametery,k=0,l=0,o=0,p=0,ftsize=15,ca=3,cr=3,mksizea=1,mksizer=1):
    '''
    Erstellt Residuenplot anhand von:
        x-Werte
            als np.array
        y-Werte
            als np.array
        Fehler(ex,ey) auf x und y
              als np.array------Falls keine x-Fehler vorliegen einfach ex als 0 angeben
        ux,uy :
                Einheiten von x und y (str) als /$ux$ angeben
        ftsize: 
                Schriftgöße
        ca:
            capsize Ausgleichsgerade
        cr:
            capsize Residuenplot
        mksize(a/r): 
            Dicke der Punkte bzw Messwert-darstellung für a/(r) Ausgleichsgraph/(Residuenplot)
        Parameterx: 
            die Abkürzung der x-Variable im Titel und auf der Achse
        Parametery: 
            die Abkürzung der y-Variable im Titel und auf der Achse
        #ru:
            Stelle auf die die Werte im Plot gerundet werden sollen
        Diese Funktion vereint lineare_regression und lineare_regression_xy der praktikumsbibliothek und plottet.
        k:
            x-position der a-Werte im Plot
        l: 
            y-Position der a-Werte im Plot
        o:
            x-Position der chiq-Werte
        p:
            y-Position der chiq-Werte
    ''' 
    
    if type(ex)==int and ex==0:
         ex=0
         #print(1)
         #k=np.min(x)
         #l=np.max(y)
         a=lineare_regression(x,y,ey)[0]
         ea=lineare_regression(x,y,ey)[1]
         b=lineare_regression(x,y,ey)[2]
         eb=lineare_regression(x,y,ey)[3]
         chiq=lineare_regression(x,y,ey)[4]
         chiq_ndof=chiq/(len(x)-2)
         Ausgleichsgerade=a*x+b
         #y-Achse vom Endplot
         res=y-Ausgleichsgerade
         #Fehlerresiduenplot
         sres=np.sqrt(ey**2+(a*ex)**2)
         """bestimmung y-Position des chiq/ndof im residuenplot"""
         #hilfsvariablen für Text im Plot
         h='a='+('{0:9.0f}').format(a)+'+/-'+('{0:9.0f}').format(ea)
         i='b='+('{0:9.4f}').format(b)+'+/-'+('{0:9.4f}').format(eb)
         j='chiq/ndof='+('{0:9.2f}').format(chiq_ndof)
         fig1, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
         ax0.errorbar(x, y,ey,0, 'o', ms = mksizea, capsize=ca)
         ax0.plot(x, Ausgleichsgerade , "r")
         ax0.set_title(Parametery +' gegen '+Parameterx +' (blau) und Ausgleichsgerade (rot)' , fontsize = ftsize)
         ax0.grid()
         ax0.annotate('{0} \n{1}'.format(h,i),xy=(k,l),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})
         ax0.set_ylabel(Parametery+ uy, fontsize = ftsize)
         ax1.set_xlabel(Parameterx+ ux, fontsize = ftsize)
         ax1.set_ylabel(Parametery+" - (a*" + Parameterx +" + b)  " + uy, fontsize = ftsize)
         ax1.errorbar(x, res, sres, 0, "o", ms = mksizer, capsize=cr)
         ax1.plot(x, 0*x, "red")
         ax1.annotate(j,xy=(o,p),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})
         ax1.set_title("Residuenplot", fontsize = ftsize)
         ax1.grid()
         #print('Steigung',a,'+/-',ea)
         #print('Achsenabschnitt',b,'+/-',eb)
         #print('chiq',chiq_ndof)
         return(a,ea,b,eb,chiq_ndof)
    elif type(ex)==np.ndarray:
        #print(2)
        ex=ex
        """bestimmung der Koordinaten für die Texte im Plot"""
        a=lineare_regression_xy(x,y,ex,ey)[0]
        ea=lineare_regression_xy(x,y,ex,ey)[1]
        b=lineare_regression_xy(x,y,ex,ey)[2]
        eb=lineare_regression_xy(x,y,ex,ey)[3]
        chiq=lineare_regression_xy(x,y,ex,ey)[4]
        chiq_ndof=chiq/(len(x)-2)
        Ausgleichsgerade=a*x+b
        #y-Achse vom Endplot
        res=y-Ausgleichsgerade
        #hilfsvariablen für Text im Plot
        h='a='+('{0:9.4f}').format(a)+'+/-'+('{0:9.4f}').format(ea)
        i='b='+('{0:9.4f}').format(b)+'+/-'+('{0:9.4f}').format(eb)
        j='chiq/ndof='+('{0:9.4f}').format(chiq_ndof)
        #Fehlerresiduenplot
        sres=np.sqrt(ey**2+(a*ex)**2)
        """bestimmung y-Position des chiq/ndof im residuenplot"""
        fig1, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        ax0.errorbar(x, y, ey, ex, 'o', ms = mksizea, capsize=ca) 
        ax0.plot(x, Ausgleichsgerade , "r")
        ax0.set_title(Parametery +' gegen '+Parameterx +' (blau) und Ausgleichsgerade (rot)' , fontsize = ftsize)
        ax0.grid()
        ax0.annotate('{0} \n{1}'.format(h,i),xy=(k,l),fontsize=15,bbox={'facecolor':'white','alpha':0.5,'pad':4})
        ax0.set_ylabel(Parametery+ uy, fontsize = ftsize)
        ax1.set_xlabel(Parameterx+ ux, fontsize = ftsize)
        ax1.set_ylabel(Parametery+" - (a*" + Parameterx +" + b)"+  uy, fontsize = ftsize)
        ax1.errorbar(x, res, sres, 0, "o", ms = mksizer, capsize=cr)
        ax1.plot(x, 0*x,"red")
        ax1.annotate(j,xy=(o,p),fontsize=15,bbox={'facecolor':'white','alpha':0.5,'pad':4})
        ax1.set_title("Residuenplot", fontsize = ftsize)
        ax1.grid()
        #verhindern des überschneidens durch erhöhen von hspace bei ausführung mit python console nicht notwendig
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        #print("Steigung",a,"+/-",ea)
        #print('Achsenabschnitt',b,'+/-',eb)
        #print('chiq/ndof',chiq_ndof)
        return(a,ea,b,eb,chiq_ndof)
        
    else:
        print('incorrect data type error x')
            
def Abweichung(x,ex,y,ey):
    '''
    Bestimmt die Abweichung eines fehlerbehafteten Wertes vom Theoriewert in Standardabweichungen
    '''
    delta=abs(x-y)/(np.sqrt(ex**2+ey**2))
    print('Abweichung',delta,'/sigma')
    return delta




def fehlerfortpflanzung(values,error,power,name=''):
    '''
    Hier übergibt man drei arrays. Das erste für die Werte der Größen, das Zweite
    für die Unsicherheiten und eins für die jeweiligen Potenzen mit denen die 
    Größe in der Formel auftritt.
           
    Benutzbar für Formeln der Art: f= a**pa*b**pb*c**pc*d**pd
    '''
    while len(values)<4:
        values+=[0]
        error+=[0]
        power+=[0]
     
    a=values[0]
    b=values[1]
    c=values[2]
    d=values[3]
    ea=error[0]
    eb=error[1]
    ec=error[2]
    ed=error[3]
    pa=power[0]
    pb=power[1]
    pc=power[2]
    pd=power[3]
    
    f= a**pa*b**pb*c**pc*d**pd
    if pd!=0:
        errorf=np.sqrt((b**pb*c**pc*d**pd*a**(pa-1)*pa*ea)**2
                       +(a**pa*c**pc*d**pd*b**(pb-1)*pb*eb)**2
                        +(a**pa*c**pc*d**pd*c**(pc-1)*pc*ec)**2
                         +(b**pb*c**pc*a**pa*d**(pd-1)*pd*ed)**2)
    elif pd==0 and pc!=0:
        errorf=np.sqrt((b**pb*c**pc*d**pd*a**(pa-1)*pa*ea)**2
                       +(a**pa*c**pc*d**pd*b**(pb-1)*pb*eb)**2
                        +(a**pa*c**pc*d**pd*c**(pc-1)*pc*ec)**2)
    elif pc==0 and pd==0 and pb!=0:
        errorf=np.sqrt((b**pb*c**pc*d**pd*a**(pa-1)*pa*ea)**2
                       +(a**pa*c**pc*d**pd*b**(pb-1)*pb*eb)**2)
    elif pc==0 and pd==0 and pb==0:
        errorf=np.sqrt((b**pb*c**pc*d**pd*a**(pa-1)*pa*ea)**2)
    #print(name +'=' +('{0:9.4f}').format(f) + '+/-' + ('{0:9.4f}').format(errorf))
    return(f,errorf)



def uncertainty_sum(values,error,power,anzahl_summanden,anzahl_faktoren_prosummand):
    '''
    hier übergibt die array values, error und power der vierte Parameter
    beschreibt die Ausgangsformel durch angabe der Anzahl an vorhandenen
    Summanden bestehen aus Produkten. Das nächste array präzisiert die
    Angabenzu jedem Summanden in dem die Anzahl der auftretenden Faktoren
    gegeben wird.
    '''
    real_value=0
    var=0
    
    #values=[2,1,1,1,2,1,1,1,2]
    #error=[1,1,1,1,1,1,1,1,1]
    #power=[1,1,1,1,1,1,1,1,1]
    #anzahl_summanden=3
    #anzahl_faktoren_prosummand=[3,3,3]
    new_values=[]
    new_error=[]
    new_power=[]
        #trennen der werte nach summanden
    if sum(anzahl_faktoren_prosummand)==len(values):
        for i in range(anzahl_summanden):
             
            new_values+=[0]
            if i==0:
                new_values[i]=[values[0:anzahl_faktoren_prosummand[i]]]
            else:
                summe=0
                for x in range(i):
                    summe+=anzahl_faktoren_prosummand[i-x-1]
                
                new_values[i]=[values[summe : summe + anzahl_faktoren_prosummand[i]]]
            
            new_error+=[0]
            if i==0:
                new_error[i]=[error[0:anzahl_faktoren_prosummand[i]]]
            else:
                summe=0
                for x in range(i):
                    summe+=anzahl_faktoren_prosummand[i-x-1]
                
                new_error[i]=[error[summe : summe + anzahl_faktoren_prosummand[i]]]    
            
            new_power+=[0]
            if i==0:
                new_power[i]=[power[0:anzahl_faktoren_prosummand[i]]]
            else:
                summe=0
                for x in range(i):
                    summe+=anzahl_faktoren_prosummand[i-x-1]
                
                new_power[i]=[power[summe : summe + anzahl_faktoren_prosummand[i]]]
            #erstellen der Varianz als summe der einzel Varianzen             
            var+=(fehlerfortpflanzung(new_values[i][0],new_error[i][0],new_power[i][0])[1])**2
            #Gesamt mittelwert als summe der Mittelwerte     
            real_value+=fehlerfortpflanzung(new_values[i][0],new_error[i][0],new_power[i][0])[0]
    else:
        print('dim(values) and sum(factors) are not equal')  
    uncertainty=np.sqrt(var)
    return(real_value, uncertainty)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    