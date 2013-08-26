#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure4 : optimization de parameters

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import *
from Sweep import Sweep_performances
from matplotlib import *
from pylab import *

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the pickle to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
correlation = 'S'


# -----------------------------------
# DATA loading
# -----------------------------------
pkfile = open(options.input,'rb')
data = pickle.load(pkfile)

label = data['label']
xs = data['xs']
ys = data['ys']

# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
#rcParams.update(params)
tit = ['B-WM', 'K-QL']
fig = figure(figsize=(9,6))
step = 2
t = "JSD"
for m,i in zip(['bmw', 'kalman'], [1,2]):
    subplot(2,2,i)
    im = imshow(data[t][m], origin = 'lower', cmap = cm.binary, interpolation = 'nearest', aspect = 'auto')
    if m == 'kalman':
        step = 3
        yticks(range(0, len(xs[m]), step), np.around(xs[m],2)[range(0, len(xs[m]), step)])
        step = 4
        xticks(range(0, len(ys[m]), step), np.around(ys[m],2)[range(0, len(ys[m]), step)])
    elif m == 'bmw':
        step = 3
        yticks(range(0, len(xs[m]), step), np.around(xs[m],2)[range(0, len(xs[m]), step)])
        step = 4
        xticks(range(0, len(ys[m]), step), np.around(ys[m],2)[range(0, len(ys[m]), step)])    
    title(tit[i-1])
    ylabel(label[m][0])
    xlabel(label[m][1])
    
cbar_ax = fig.add_axes([0.83, 0.6, 0.02, 0.25])
figtext(0.79,0.73, t, fontsize = 13)
colorbar(im, cax=cbar_ax)

t = "Z"
for m,i in zip(['bmw', 'kalman'], [3,4]):
    subplot(2,2,i)
    im2 = imshow(data[t][m], origin = 'lower', cmap = cm.binary, interpolation = 'nearest', aspect = 'auto')
    if m == 'kalman':
        step = 3
        yticks(range(0, len(xs[m]), step), np.around(xs[m],2)[range(0, len(xs[m]), step)])
        step = 4
        xticks(range(0, len(ys[m]), step), np.around(ys[m],2)[range(0, len(ys[m]), step)])
    elif m == 'bmw':
        step = 3
        yticks(range(0, len(xs[m]), step), np.around(xs[m],2)[range(0, len(xs[m]), step)])
        step = 4
        xticks(range(0, len(ys[m]), step), np.around(ys[m],2)[range(0, len(ys[m]), step)])    
    ylabel(label[m][0])
    xlabel(label[m][1])
    
cbar_ax = fig.add_axes([0.83, 0.14, 0.02, 0.25])
figtext(0.81,0.25, t, fontsize = 13)
colorbar(im2, cax=cbar_ax)

figtext(0.085, 0.90, "A", fontsize = 18)
figtext(0.45, 0.90, "B", fontsize = 18)
figtext(0.085, 0.43, "C", fontsize = 18)
figtext(0.45, 0.43, "D", fontsize = 18)

#colorbar(cax, shrink=0.5, pad=.2, aspect=10)
#colorbar(cax)
subplots_adjust(wspace = 0.45, hspace = 0.4, right = 0.75)
#fig.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig4.pdf', bbox_inches='tight')
show()




