#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""

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
import pickle
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import leastsq
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------


# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y : (y - fitfunc(p, x))

def leastSquares(x, y):
    for i in xrange(len(x)):
        pinit = [1.0, -1.0]
        p = leastsq(errfunc, pinit, args = (x[i], y[i]), full_output = False)
        x[i] = fitfunc(p[0], x[i])
    return x    

def center(x):
    #x = x-np.mean(x)
    #x = x/np.std(x)
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x

def mutualInformation(x, y):
    np.seterr('ignore')
    bin_size = 2*(np.percentile(y, 75)-np.percentile(y, 25))*np.power(len(y), -(1/3.))
    py, edges = np.histogram(y, bins=np.arange(y.min(), y.max()+bin_size, bin_size))
    py = py/float(py.sum())
    yp = np.digitize(y, edges)-1
    px, edges = np.histogram(x, bins = np.linspace(x.min(), x.max()+0.00001, 25))
    px = px/float(px.sum())
    xp = np.digitize(x, edges)-1
    p = np.zeros((len(py), len(px)))
    for i in xrange(len(yp)): p[yp[i], xp[i]] += 1
    p = p/float(p.sum())
    tmp = np.log2(p/np.outer(py, px))
    tmp[np.isinf(tmp)] = 0.0
    tmp[np.isnan(tmp)] = 0.0
    return (np.sum(p*tmp), p)



# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_repets = 4
nb_blocs = 4
nb_trials = 39
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
p_test = eval(open('parameters.txt', 'r').read())
#keys = ['S13','S9','S8']
#keys = ['S13']
keys = ['S13', 'S9', 'S8', 'S2', 'S11', 'S17', 'S19', 'S5', 'S6', 'S20', 'S15', 'S12', 'S14', 'S16']

hrt = []
hrtm = []
mi = []
pmi = []

#for s in p_test.iterkeys():
for s in keys:
    m = p_test[s].keys()[0]
    models[m].setAllParameters(p_test[s][m])
    models[m].startExp()
    for k in xrange(nb_repets):
        for i in xrange(nb_blocs):
            cats.reinitialize()
            cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
            models[m].startBloc()
            for j in xrange(nb_trials):
                state = cats.getStimulus(j)
                action = models[m].chooseAction(state)
                reward = cats.getOutcome(state, action)
                models[m].updateValue(reward)
    rtm = np.array(models[m].reaction)
    state = convertStimulus(np.array(models[m].state))
    action = np.array(models[m].action)
    responses = np.array(models[m].responses)    
    step, indice = getRepresentativeSteps(rtm, state, action, responses)
    hrtm.append(computeMeanRepresentativeSteps(step)[0])

    
    rt = np.array([human.subject['fmri'][s][i]['rt'][0:nb_trials,0] for i in range(1,nb_blocs+1)])    
    rt = np.tile(rt, (nb_repets,1))
    action = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,1] for i in range(1,nb_blocs+1)])
    responses = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,2] for i in range(1,nb_blocs+1)])    
    action = np.tile(action, (nb_repets, 1))
    responses = np.tile(responses, (nb_repets, 1))
    step, indice2 = getRepresentativeSteps(rt, state, action, responses)
    hrt.append(computeMeanRepresentativeSteps(step)[0])

hrt = np.array(hrt)
hrtm = np.array(hrtm)
mi = np.array(mi)
pmi = np.array(pmi)


hrt = np.array(map(center, hrt))
hrtm = np.array(map(center, hrtm))
#hrtm = leastSquares(hrtm, hrt)

fig = figure(figsize = (15, 12))

ax1 = fig.add_subplot(4,4,1)
ax1.plot(np.mean(hrt, 0), 'o-')
#ax2 = ax1.twinx()
ax1.plot(np.mean(hrtm, 0), 'o-', color = 'green')

for i, s in zip(xrange(14), keys):
  ax1 = fig.add_subplot(4,4,i+2)
  ax1.plot(hrt[i], 'o-')
  #ax2 = ax1.twinx()
  ax1.plot(hrtm[i], 'o--', color = 'green')
  ax1.set_title(s)



# fig3 = figure(figsize = (15, 12))
# for i, s in zip(xrange(14), p_test.keys()):
#     subplot(4,4,i+1)
#     imshow(pmi[i], origin = 'lower', interpolation = 'nearest')
#     title(s + " " + str(mi[i]))

show()


sys.exit()













# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
rt = computeMeanRepresentativeSteps(step)
step, indice = getRepresentativeSteps(model.responses, model.state, model.action, model.responses)
y = computeMeanRepresentativeSteps(step)
distance = computeDistanceMatrix(model.state, indice)

correct = np.array([model.reaction[np.where((distance == i) & (model.responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
incorrect = np.array([model.reaction[np.where((distance == i) & (model.responses  == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_correct = np.array([np.mean(model.reaction[np.where((distance == i) & (model.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_correct = np.array([sem(model.reaction[np.where((distance == i) & (model.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
mean_incorrect = np.array([np.mean(model.reaction[np.where((distance == i) & (model.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_incorrect = np.array([sem(model.reaction[np.where((distance == i) & (model.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
indice_fmri = indice
y_fmri = computeMeanRepresentativeSteps(step)
distance_fmri = computeDistanceMatrix(human.stimulus['fmri'], indice)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
y_fmri = computeMeanRepresentativeSteps(step)



# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure(figsize = (9,4))
ion()
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          
#rcParams.update(params)                  
colors = ['blue', 'red', 'green']
subplot(1,2,1)
for i in xrange(3):
    plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
    #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)
    ylabel("Probability correct responses")
    legend(loc = 'lower right')
    xticks(range(2,len(pcr['mean'][i])+1,2))
    xlabel("Trial")
    xlim(0.8, len(pcr['mean'][i])+1.02)
    ylim(-0.05, 1.05)
    yticks(np.arange(0, 1.2, 0.2))
    title('A')
    grid()


ax1 = plt.subplot(1,2,2)
ax1.plot(range(1, len(rt_fmri[0])+1), rt_fmri[0]-0.2, linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0]-0.2, rt_fmri[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
ax2.set_ylim(-4, 12)
##
msize = 8.0
mwidth = 2.5
ax1.plot(1, 0.455, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
ax1.plot(1, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(1, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(2, 0.455, 'o', color = 'blue', markersize=msize)
ax1.plot(2, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(2, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(3, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(3, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(4, 0.4445, 'o', color = 'red', markersize=msize)
ax1.plot(4, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(5, 0.435, 'o', color = 'green', markersize=msize)
for i in xrange(6,16,1):
    ax1.plot(i, 0.455, 'o', color = 'blue', markersize=msize)
    ax1.plot(i, 0.4445, 'o', color = 'red', markersize=msize)
    ax1.plot(i, 0.435, 'o', color = 'green', markersize=msize)

##
ax1.set_ylabel("Reaction time (s)")
ax1.grid()
ax1.set_xlabel("Representative steps")
ax1.set_xticks([1,5,10,15])
#ax1.set_yticks([0.46, 0.50, 0.54])
#ax1.set_ylim(0.43, 0.56)
ax1.set_title('B')

################

# ind = np.arange(1, len(rt[0])+1)
# ax5 = subplot(2,2,3)
# for i,j,k,l,m in zip([y, y_meg, y_fmri], 
#                    ['blue', 'grey', 'grey'], 
#                    ['model', 'MEG', 'FMRI'],
#                    [1.0, 0.9, 0.9], 
#                    ['-', '--', ':']):
#     ax5.plot(ind, i[0], linewidth = 2, color = j, label = k, alpha = l, linestyle = m)
#     ax5.errorbar(ind, i[0], i[1], linewidth = 2, color = j, alpha = l, linestyle = m)

# ax5.grid()
# ax5.set_ylabel("PCR %")    
# ax5.set_yticks(np.arange(0, 1.2, 0.2))
# ax5.set_xticks(range(2, 15, 2))
# ax5.set_ylim(-0.05, 1.05)
# ax5.legend(loc = 'lower right')


################
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)
#savefig('../../../Dropbox/ISIR/B2V_council/images/fig_subject'+options.model+'.pdf', bbox_inches='tight')
savefig('test.pdf', bbox_inches='tight')









