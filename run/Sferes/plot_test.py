#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject
for sferes

run subjectTest.py -i sferes_fmri.txt -m kalman

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from Selection import *
from matplotlib import *
from pylab import *

import matplotlib.pyplot as plt
from time import time
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the parameters file to load", default=False)

(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))


# ------------------------------------
# Model Loading
# ------------------------------------
data = loadData(options.input)
pcr = data['pcr']
rt = data['rt']

# -----------------------------------
# FMRI
# -----------------------------------
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_human = computeMeanRepresentativeSteps(step) 
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
colors = ['blue', 'red', 'green']
fig = figure()
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['backend'] = 'pdf'
rcParams['figure.figsize'] = 7.3,4.2

# Performance
ax1 = fig.add_subplot(2,2,1)
[errorbar(range(1, len(pcr_human['mean'][t])+1), pcr_human['mean'][t], pcr_human['sem'][t], linewidth = 2.5, elinewidth = 1.5, capsize = 0.8, linestyle = '--', color = colors[t], alpha = 0.7) for t in xrange(3)]    
for m in ['qlearning', 'bayesian']:
    [errorbar(range(1, len(pcr[m]['mean'][t])+1), pcr[m]['mean'][t], pcr[m]['sem'][t], linewidth = 1.5, elinewidth = 1.5, capsize = 0.8, linestyle = '-', color = colors[t], alpha = 1) for t in xrange(3)]    
ax1.set_xticks(range(2,len(pcr_human['mean'][0])+1,2))
ax1.set_xlabel("Trial")
ax1.set_xlim(0.8, len(pcr_human['mean'][0])+1.02)
ax1.set_ylabel("Probability correct responses") 
ax1.set_ylim(-0.05, 1.05)    
ax1.set_yticks([0,0.25,0.5,0.75,1.0])
ax1.grid()
#ax.yaxis.set_major_locator(MaxNLocator(4))

ax2 = fig.add_subplot(2,2,3)
[errorbar(range(1, len(pcr_human['mean'][t])+1), pcr_human['mean'][t], pcr_human['sem'][t], linewidth = 2.5, elinewidth = 1.5, capsize = 0.8, linestyle = '--', color = colors[t], alpha = 0.7) for t in xrange(3)]    
for m in ['fusion']:
    [errorbar(range(1, len(pcr[m]['mean'][t])+1), pcr[m]['mean'][t], pcr[m]['sem'][t], linewidth = 1.5, elinewidth = 1.5, capsize = 0.8, linestyle = '-', color = colors[t], alpha = 1) for t in xrange(3)]    
ax2.set_xticks(range(2,len(pcr_human['mean'][0])+1,2))
ax2.set_xlabel("Trial")
ax2.set_xlim(0.8, len(pcr_human['mean'][0])+1.02)
ax2.set_ylabel("Probability correct responses") 
ax2.set_ylim(-0.05, 1.05)    
ax2.set_yticks([0,0.25,0.5,0.75,1.0])
ax2.grid()
#ax.yaxis.set_major_locator(MaxNLocator(4))

# REACTION TIME
ax3 = fig.add_subplot(2,2,2)
ax3.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
ax4 = ax3.twinx()
for m in ['qlearning','bayesian']:
    ax4.errorbar(range(1, len(rt[m][0])+1), rt[m][0], rt[m][1], linewidth = 2.0, elinewidth = 1.5, capsize = 1.0, linestyle = '-', color = 'black', alpha = 1.0)
ax3.grid()
ax3.xaxis.set_major_locator(MaxNLocator(5))
ax3.set_ylabel("Reaction time (s)")
ax3.set_xlabel("Representative step")
ax3.yaxis.set_major_locator(MaxNLocator(4))
ax4.yaxis.set_major_locator(MaxNLocator(4))


ax5 = fig.add_subplot(2,2,4)
ax5.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
ax6 = ax5.twinx()
for m in ['fusion']:
    ax6.errorbar(range(1, len(rt[m][0])+1), rt[m][0], rt[m][1], linewidth = 2.0, elinewidth = 1.5, capsize = 1.0, linestyle = '-', color = 'black', alpha = 1.0)
ax5.grid()
ax5.xaxis.set_major_locator(MaxNLocator(5))
ax5.set_ylabel("Reaction time (s)")
ax5.set_xlabel("Representative step")
ax5.yaxis.set_major_locator(MaxNLocator(4))
ax6.yaxis.set_major_locator(MaxNLocator(4))


# SAVING PLOT
fig.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
#fig.tight_layout(pad = 1.3)
#fig.show(
fig.savefig('test.pdf')
os.system("evince test.pdf")

