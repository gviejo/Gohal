#!/usr/bin/python
# encoding: utf-8
"""
KLearning.py

performs a Kelman Qlearning over CATS task

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../Plot")
from plot import plotCATS
sys.path.append("../LearningAnalysis")
from LearningAnalysis import SSLearning
sys.path.append("../Fonctions")
from fonctions import *
import scipy.io as sio
from ColorAssociationTasks import CATS
from pylab import *
from scipy import stats

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="Load reaction time data 'reaction_time.dat'", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
# -----------------------------------

# -----------------------------------
# Loading meg data
# -----------------------------------
meg = loadDirectoryEEG('PEPS_GoHaL/Beh_Model/')
megreaction = sio.loadmat('PEPS_GoHaL/reaction_timeMEG.mat')
megresponses = []
for i in meg.keys():
    for j in xrange(1, 5):
        megresponses.append(meg[i][j]['sar'][:,2][0:40])
megresponses = np.array(megresponses)
# -----------------------------------

# -----------------------------------
# Loading fmri data
# -----------------------------------
fmri = loadDirectoryfMRI('fMRI')
fmrireaction = computeMeanReactionTime(fmri, case = 'fmri')
fmriresponses = []
for i in fmri.iterkeys():
    for j in fmri[i].iterkeys():
        fmriresponses.append(fmri[i][j]['sar'][:,2][0:40])
fmriresponses = np.array(fmriresponses)
# -----------------------------------



# -----------------------------------
#Plot 
# -----------------------------------
figure()
subplot(211)
m = np.mean(megresponses, 0)
s = stats.sem(megresponses, 0)
plot(m, linewidth = 2, label = 'Meg')
fill_between(range(40), m+s, m-s, alpha = 0.2)
m = np.mean(fmriresponses, 0)
s = stats.sem(fmriresponses, 0)
plot(m, linewidth = 2, label = 'Fmri')
fill_between(range(40), m+s, m-s, alpha = 0.2)
legend()
grid()
title('Accuracy')

subplot(212)
m = megreaction['mean'][0:40].flatten()
s = megreaction['sem'][0:40].flatten()
plot(m, linewidth = 2, label = 'Meg')
fill_between(range(40), m+s, m-s, alpha = 0.2)
m = fmrireaction['mean'][0:40]
s = fmrireaction['sem'][0:40]
plot(m, linewidth = 2, label = 'Fmri')
fill_between(range(40), m+s, m-s, alpha = 0.2)
title('Reaction time')
legend()
grid()

show()
