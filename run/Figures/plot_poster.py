#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
def center(x):
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
tmp = np.reshape(human.reaction['fmri'], (14, 4*39))
tmp = np.array(map(center, tmp))
human.reaction['fmri'] = np.reshape(tmp, (14*4, 39))
# -----------------------------------

# -----------------------------------
# LOADING DATA
# -----------------------------------
with open("beh_model.pickle", 'r') as handle:
    data = pickle.load(handle)

# -----------------------------------
#order data
# -----------------------------------
hpcr = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
mpcr = data['pcr']
# -----------------------------------
step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
hrt = computeMeanRepresentativeSteps(step)
mrt = data['rt']
# -----------------------------------
# Subject to show
subjects = ['S9', 'S5']

# -----------------------------------
# Plot
# -----------------------------------

fig = plt.figure(figsize = (16, 8))

dashes = ['-', '--', ':']
colors = ['blue','red','green']
line1 = tuple([plt.Line2D(range(1),range(1),marker='o',alpha=1.0,color=colors[i]) for i in xrange(3)])
plt.figlegend(line1,tuple(["Stim 1", "Stim 2", "Stim 3"]), loc = 'lower right', bbox_to_anchor = (0.8, 0.73))

ax1 = fig.add_subplot(221)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

for i in xrange(3):
    plot(range(1, len(mpcr['mean'][i])+1), mpcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(mpcr['mean'][i])+1), mpcr['mean'][i], mpcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(hpcr['mean'][i])+1), hpcr['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    


ax1.set_ylabel("Probability correct responses")
ax1.set_xlabel("Trial")
fig.text(0.1, 0.95, "A.", fontsize = 22)
# legend(loc = 'lower right')
# xticks(range(2,11,2))
# xlabel("Trial")
# xlim(0.8, 10.2)
# ylim(-0.05, 1.05)
# yticks(np.arange(0, 1.2, 0.2))
# ylabel('Probability Correct Responses')
# title('A')

ax2 = fig.add_subplot(222)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()


ax2.errorbar(range(1, len(hrt[0])+1), hrt[0], hrt[1], linewidth = 2, color = 'grey', alpha = 0.5)
ax2.errorbar(range(1, len(mrt[0])+1), mrt[0], mrt[1], linewidth = 2, color = 'black', alpha = 0.9)

ax2.set_xlabel("Representative step")
ax2.set_ylabel("Reaction time (s)")

# ###
fig.text(0.52, 0.95, "B.", fontsize = 22)

ax3 = fig.add_subplot(223)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()

ax3.plot(range(1, len(data['s']['h']['S9'])), data['s']['h']['S9'], linewidth = 2, color = 'grey', alpha = 0.7)
ax3.plot(range(1, len(data['s']['m']['S9'])), data['s']['m']['S9'], linewidth = 2, color = 'black', alpha = 0.7)

ax4 = fig.add_subplot(224)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.get_xaxis().tick_bottom()
ax4.get_yaxis().tick_left()

ax4.plot(range(1, len(data['s']['h']['S5'])), data['s']['h']['S5'], linewidth = 2, color = 'grey', alpha = 1.0)
ax4.plot(range(1, len(data['s']['m']['S5'])), data['s']['m']['S5'], linewidth = 2, color = 'black', alpha = 1.0)


# subplots_adjust(hspace = 0.2, left = 0.2)
fig.savefig(os.path.expanduser("~/Dropbox/ISIR/SBDM/poster_2014/pics/beh_model.eps"), bbox_inches='tight')
os.system("evince "+os.path.expanduser("~/Dropbox/ISIR/SBDM/poster_2014/pics/beh_model.eps"))


