#!/usr/bin/python
# encoding: utf-8
"""
Test for Fusion Selection


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Selection import FSelection
from Sferes import RBM
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time


# -----------------------------------
# FONCTIONS
# -----------------------------------
def learnRBM():
    pdf = np.array(model.pdf).reshape(14,156,8)
    rtm = model.reaction.reshape(14,156)
    new_rtm = []
    for i in xrange(len(rt)):
        bin_size = 2*(np.percentile(rt[i], 75)-np.percentile(rt[i], 25))*np.power(len(rt[i]), -(1/3.))
        mass, edges = np.histogram(rt[i], bins=np.arange(rt[i].min(), rt[i].max()+bin_size, bin_size))
        mass = mass/float(mass.sum())
        position = np.digitize(rt[i], edges)-1
        dataset = np.zeros((len(rt[i]), position.max()+1))
        for j in xrange(len(position)): dataset[j,position[j]] = 1.0
        rbm = RBM(dataset, pdf[i], stop=0.0001,epsilon=0.001)
        rbm.train()        
        rbm.getInputfromOutput(pdf[i])
        #tmp = np.zeros((len(rtm[i]),pdf.shape[2]))        
        #for j in xrange(len(rtm[i])): tmp[j,rtm[i][j]] = 1.0
        #rbm.x = np.zeros(rbm.x.shape)
        #rbm.x[:,rbm.nx:] = tmp
        
        #p = rbm.xx[:,0:rbm.nx]        
        #p = np.exp(p*2.0)/np.vstack(np.exp(p*2.0).sum(1))        
        tirage = np.argmax(rbm.xx[:,0:rbm.nx], 1)
        center = edges[1:]-(bin_size/2.)
        
        new_rtm.append(center[tirage])
        
    return np.array(new_rtm).reshape(56,39)


def testModel():
    model.startExp()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        model.startBloc()
        for j in xrange(nb_trials):
            sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()                    
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)
            model.updateValue(reward)

    model.state = convertStimulus(np.array(model.state))
    model.action = convertAction(np.array(model.action))
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)
    


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
rt = human.reaction['fmri'].reshape(14, 156)
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
very_good_parameters = dict({'noise':0.0001,
                    'length':10,
                    'alpha':0.8,
                    'beta':3.0,
                    'gamma':0.4,
                    'threshold':4.0,
                    'gain':2.0})
parameters = dict({'noise':0.2,
                    'length':7,
                    'alpha':0.9,
                    'beta':3.5,
                    'gamma':0.5,
                    'gain':0.6,
                    'threshold':1.5})
              

nb_trials = 39
nb_blocs = 56
cats = CATS(nb_trials)

model = FSelection(cats.states, cats.actions, parameters)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
t1 = time()
testModel()


rtm = learnRBM()
model.reaction = rtm

t2 = time()



print "\n"
print t2-t1
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

# thr = extractStimulusPresentation(model.thr, model.state, model.action, model.responses)
# thr_free = extractStimulusPresentation(model.thr_free, model.state, model.action, model.responses)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 

step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
rt = computeMeanRepresentativeSteps(step)

# step, indice = getRepresentativeSteps(model.thr, model.state, model.action, model.responses)
# thr_step = computeMeanRepresentativeSteps(step) 

# step, indice = getRepresentativeSteps(model.thr_free, model.state, model.action, model.responses)
# thr_free_step = computeMeanRepresentativeSteps(step) 


# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (9,7))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          
#rcParams.update(params)                  
colors = ['blue', 'red', 'green']
subplot(2,2,1)
for i in xrange(3):
    plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
    #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)
    ylabel("Probability correct responses")
    legend(loc = 'lower right')
    xticks(range(2,11,2))
    xlabel("Trial")
    xlim(0.8, 10.2)
    ylim(-0.05, 1.05)
    yticks(np.arange(0, 1.2, 0.2))
    title('A')
    grid()


ax1 = plt.subplot(2,2,2)
ax1.plot(range(1, len(rt_fmri[0])+1), rt_fmri[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
#x2.set_ylim(-5, 15)
ax1.grid()
############

# subplot(2,2,3)
# for i in xrange(3):
#     plot(range(1, len(thr['mean'][i])+1), thr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
#     errorbar(range(1, len(thr['mean'][i])+1), thr['mean'][i], thr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
#     plot(range(1, len(thr_free['mean'][i])+1), thr_free['mean'][i], linewidth = 2, linestyle = '--', color = colors[i], label= 'Stim '+str(i+1))    
#     errorbar(range(1, len(thr_free['mean'][i])+1), thr_free['mean'][i], thr_free['sem'][i], linewidth = 2, linestyle = '--', color = colors[i])
#     ylabel("H(p(r/s))")
#     xticks(range(2,11,2))
#     xlabel("Trial")
#     xlim(0.8, 10.2)
#     ylim(-0.05, model.max_entropy+0.2)
#     #yticks(np.arange(0, 1.2, 0.2))    
#     grid()
# subplot(2,2,4)
# plot(range(1, len(thr_step[0])+1), thr_step[0], linewidth = 2, linestyle = '-', color = 'black', alpha = 0.9)
# errorbar(range(1, len(thr_step[0])+1), thr_step[0], thr_step[1], linewidth = 2, linestyle = '-', color = 'black', alpha = 0.9)
# plot(range(1, len(thr_free_step[0])+1), thr_free_step[0], linewidth = 2, linestyle = '-', color = 'grey', alpha = 0.9)
# errorbar(range(1, len(thr_free_step[0])+1), thr_free_step[0], thr_free_step[1], linewidth = 2, linestyle = '-', color = 'grey', alpha = 0.9)
# ylim(-0.05, model.max_entropy+0.2)
# ylabel("H(p(r/s))")
# xlabel("Representative step")
# grid()




subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testSelection.pdf', bbox_inches='tight')
#savefig('/home/viejo/Desktop/figure_guillaume_a_tord.pdf', bbox_inches='tight')
show()
