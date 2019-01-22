
### PraktLib  

# Benutzung als Klasse:
#     
#     p = PraktLib (string,string)    # erzeugt eine Instanz von Praktlib (Beispiel: PraktLib("filename.lab","cassy") )
#     p.getheader()                   # liest die Spaltenüberschriften, wenn gewünscht
#     p.getcolumn(index)              # gibt spalte(index zurück), testet bounds wenn index <0 oder index >= p.columns
#     p.getdata()                     # gibt das datenarray zurück
# 
#     p.data                          # greift direkt auf das Datenarray zu
#     p.rows                          # gibt Zahl der Zeilen
#     p.columns                       # gibt Zahl der Spalten
# 

# In[1]:

# import SciPy  numpy package
import numpy as np
import re
import sys
import os
import copy
class PraktLib(object) :
    
    def __init__(self,name,mode) :
        self.columns = 0
        self.rows = 0
        if mode != "cassy" :
            print "only Cassy Lab 1 data files implemented"
            return None
        f = open(name, 'r')
        line = f.readline()
        f.close()
        if not line == 'CL4\n' :
            print name,': wrong file format - not CASSY 1'
            return None
	f = open(name, 'r')
        # Datenformat der CASSY 1 files startet mit einem Header 
        # Nur die Datenzeilen (Array von Meßwerten) enthalten \t - TABs
        # als Separator.
        # Wir suchen also die erste Zeile, die einen '\t' enthält.
        tabsgefunden = False
        i = 0
        r = 0
        # print repr(os.linesep)

        for line in f:
            if not tabsgefunden :
                if '\t' in line :
                    tabsgefunden = True;
                    previous = re.split(' ',previous)
                    previous = list(map(int,previous))
                    self.columns = previous[0]
                    r = self.rows = previous[1]
                    self.data = np.zeros((r,self.columns))
                   # print rows,'x',columns, 'zur Kontrolle: Arraydimension'
                previous = line
            if tabsgefunden :
                if '\t' in line :
                    line = re.split('[ \t]',line)
                    line.pop(-1)                # '\n' - letztes Element entfernen
                    line = list(map(float,line)) # strings nach floats konvertieren
                    for k in range(0,self.columns):
                        self.data[i,k] = line[k]
                    i = i + 1
                    r=r-1
                else :
                    break
       #  print self.rows,' Datenzeilen gelesen.' 
        f.close()
        f = open(name, 'r')
        line = f.readline()
        if not line == 'CL4\n' :
            print 'wrong file format - not CASSY 1'
            return None
        f.readline() # skip eine weitere Zeile
        i = 0
        while True :   # bis zur ersten Zeile, die ein TAB enthält, lesen
            line = f.readline()
            if line == None :
                break
            if '\t' in line :
                break
            if re.search('^[A-Za-zäöüßÄÖÜ]',line) :
                if line == None :
                    break;
                line = f.readline() # skip über zwei Nachfolgezeilen
                if line == None :
                    break;
                line = f.readline() 
                if line == None :
                    break;
                i=i+1
        f.close()
        self.header=np.empty((i, 3), dtype=object)
        f = open(name, 'r')
        i = 0
        f.readline() # überspringe CL4 und Zeile danach
        f.readline() 
        while True : # nochmal durch den File durchgehen und die Headerzeilen einsammeln
            line = f.readline()
            if line == None :
                break
            if '\t' in line :
                break
            if re.search('^[A-Za-z]',line) :
                self.header[i][0]=line.rstrip()
                self.header[i][1]=f.readline().rstrip()
                self.header[i][2]=f.readline().rstrip()
                i=i+1
        f.close()
        return None
    
    def column(self,column) :
        if column < 0 or column > self.columns - 1 :
            print "PraktLib: angefragter Spaltenindex < 0 oder > ", self.columns
        return self.data[:,column]
    
    def getheader(self) :
        return self.header
    
    def getdata(self) :
        return self.data


