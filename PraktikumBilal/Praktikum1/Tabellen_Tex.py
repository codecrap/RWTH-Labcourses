# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy as sp
import pylatex as py
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Package, Matrix
from pylatex.utils import italic
import os



if __name__ == '__main__':
    doc = Document()
    doc.packages.append(Package('geometry', options=['left=2.5cm','right=2.5cm','top=2cm','bottom=2cm']))
    doc.packages.append(Package('float'))
    with doc.create(Tabular('c|c|c|c')) as table:
        table.add_row((' ', 'Hersteller', 'Kalibration', 'Gesamt'))
        table.add_hline()
        table.add_row(('Lambda0', a[0], b[0], c[0]))
        table.add_row(('Lambda1', a[1], b[1], c[1]))
        table.add_row(('Lambda2', a[2], b[2], c[2]))
        table.add_row(('Lambda3', a[3], b[3], c[3]))
        table.add_row(('Lambda4', a[4], b[4], c[4]))
        table.add_row(('Lambda5', a[5], b[5], c[5]))
        table.add_row(('Lambda6', a[6], b[6], c[6]))
        table.add_row(('Lambda7', a[7], b[7], c[7]))
        table.add_row(('Lambda8', a[8], b[8], c[8]))
        table.add_row(('Lambda9', a[9], b[9], c[9]))
        table.add_row(('Lambda10', a[10], b[10], c[10]))
        table.add_row(('Lambda11', a[11], b[11], c[11]))
        table.add_row(('Lambda12', a[12], b[12], c[12]))
        table.add_row(('Lambda13', a[13], b[13], c[13]))
        table.add_row(('Lambda14', a[14], b[14], c[14]))
        table.add_row(('Lambda15', a[15], b[15], c[15]))
        table.add_row(('Lambda16', a[16], b[16], c[16]))
        table.add_row(('Lambda17', a[17], b[17], c[17]))
        table.add_row(('Lambda18', a[18], b[18], c[18]))
        table.add_row(('Lambda19', a[19], b[19], c[19]))
        table.add_row(('Lambda20', a[20], b[20], c[20]))
doc.generate_tex('Tabelle_1')
