#!/usr/bin/python
# encoding: utf-8
"""
to plot representative steps for meg and fmri

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *

from matplotlib import *
from pylab import *
from HumanLearning import HLearning

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (12, 8))
ion()
for case, i in zip(['meg', 'fmri'], [1,2]):    
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    rt = computeMeanRepresentativeSteps(step) 
    step, indice = getRepresentativeSteps(responses, state, action, responses)
    y = computeMeanRepresentativeSteps(step)

    ax1 = subplot(2,1,i)
    ind = np.arange(1, len(rt[0])+1)
    ax1.plot(ind, y[0], linewidth = 2, color = 'blue')
    ax1.errorbar(ind, y[0], y[1], linewidth = 2, color = 'blue')    
    ax2 = ax1.twinx()
    ax2.plot(ind, rt[0], linewidth = 2, color = 'green', linestyle = '--')
    ax2.errorbar(ind, rt[0], rt[1], linewidth = 2, color = 'green', linestyle = '--')
    ax1.grid()
    ax1.set_ylabel("PCR %")    
    ax2.set_ylabel("Reaction time (s)")
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticks(range(2, 15, 2))
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(case)
    if case == 'meg':
        ax2.set_yticks([0.46, 0.50, 0.54])
        ax2.set_ylim(0.43, 0.56)
    elif case == 'fmri':
        ax2.set_yticks([0.68, 0.72, 0.76, 0.80])
        
    

# -----------------------------------
