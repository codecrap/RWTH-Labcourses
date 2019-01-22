import numpy as np
import io

def lese_lab_datei(dateiname):
    '''
    CASSY LAB Datei einlesen (Version fuer python3).

    Messdaten werden anhand von Tabulatoren identifiziert.

    Gibt ein numpy-Array zurueck.

    '''
    f = open(dateiname)
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
    return np.genfromtxt(io.BytesIO(dnew))
