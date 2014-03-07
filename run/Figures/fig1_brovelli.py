#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure 1 : performances des sujets / RT on representatieve steps

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser
import numpy as np
from matplotlib import rc_file
rc_file("figures.rc")
import matplotlib.pyplot as plt
sys.path.append("../../src")
from fonctions import *


from HumanLearning import HLearning
from matplotlib import *
from pylab import *
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
data = dict()
data['pcr'] = dict()
for i in human.directory.keys():
    print i
    data['pcr'][i] = extractStimulusPresentation(human.responses[i], human.stimulus[i], human.action[i], human.responses[i])
# -----------------------------------
data['rt'] = dict()
for i in human.directory.keys():
    print i
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 
# -----------------------------------
data['rt2'] = dict()
for i in human.directory.keys():
    print i
    data['rt2'][i] = extractStimulusPresentation(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

fig = plt.figure()

dashes = ['-', '--', ':']
colors = ['blue','red','green']
line1 = tuple([plt.Line2D(range(1),range(1),marker='o',alpha=1.0,color=colors[i]) for i in xrange(3)])
plt.figlegend(line1,tuple(["Stim 1", "Stim 2", "Stim 3"]), loc = 'lower right', bbox_to_anchor = (0.85, 0.65))

ax1 = fig.add_subplot(211)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

m = data['pcr']['fmri']['mean']
s = data['pcr']['fmri']['sem']
for i in xrange(3):
    ax1.errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[0], color = colors[i], elinewidth = 2)
    ax1.plot(range(1, len(m[i])+1), m[i], linestyle = dashes[0], color = colors[i], linewidth = 2.5, label = 'Stim '+str(i+1))

ax1.set_ylabel("Probability correct responses")
ax1.set_xlabel("Trial")
fig.text(0.04, 0.92, "A.", fontsize = 22)
# legend(loc = 'lower right')
# xticks(range(2,11,2))
# xlabel("Trial")
# xlim(0.8, 10.2)
# ylim(-0.05, 1.05)
# yticks(np.arange(0, 1.2, 0.2))
# ylabel('Probability Correct Responses')
# title('A')

ax2 = fig.add_subplot(212)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

m = data['rt']['fmri']['mean']
s = data['rt']['fmri']['sem']
ax2.errorbar(range(1, len(m)+1), m, s, color = 'black', marker = 'o', linewidth = 2.5, elinewidth = 2)
ax2.set_xlabel("Representative step")
ax2.set_ylabel("Reaction time (s)")
# ###
msize = 11.0
mwidth = 2.5
ax2.plot(1, 0.652, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
ax2.plot(1, 0.640, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(1, 0.628, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(2, 0.652, 'o', color = 'blue', markersize=msize)
ax2.plot(2, 0.640, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(2, 0.628, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(3, 0.640, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(3, 0.628, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(4, 0.640, 'o', color = 'red', markersize=msize)
ax2.plot(4, 0.628, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(5, 0.628, 'o', color = 'green', markersize=msize)
for i in xrange(6,16,1):
    ax2.plot(i, 0.652, 'o', color = 'blue', markersize=msize)
    ax2.plot(i, 0.640, 'o', color = 'red', markersize=msize)
    ax2.plot(i, 0.628, 'o', color = 'green', markersize=msize)
ax2.set_ylim(0.62, 0.82)
# ###
fig.text(0.04, 0.46, "B.", fontsize = 22)


subplots_adjust(hspace = 0.2)
fig.savefig(os.path.expanduser("~/Dropbox/ED3C/Journee_doctorant/poster/pics/beh.eps"), bbox_inches='tight')
os.system("evince "+os.path.expanduser("~/Dropbox/ED3C/Journee_doctorant/poster/pics/beh.eps"))
plt.show()
