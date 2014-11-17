#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject

run subjectTest.py -i data

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

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))

# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 39
nb_repeat = 1000
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions),
                "mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("parameters.pickle", 'r') as f:
  p_test = pickle.load(f)

super_data = dict()
super_rt = dict()

for o in p_test.iterkeys():
# for o in ['owa']:
    hrt = []
    hrtm = []
    pcrm = dict({'s':[], 'a':[], 'r':[], 't':[]})
    rt_all = []
    rtm_all = []
    super_rt[o] = dict({'model':[]})
    for s in p_test[o].iterkeys():
        with open("fmri/"+s+".pickle", "rb") as f:
            data = pickle.load(f)          
        m = p_test[o][s].keys()[0]
        print "Testing "+s+" with "+m+" selected by "+o
        models[m].setAllParameters(p_test[o][s][m])
        models[m].startExp()
        for k in xrange(nb_repeat):
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
                models[m].startBloc()
                for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = models[m].chooseAction(state)
                    reward = cats.getOutcome(state, action, case='fmri')
                    models[m].updateValue(reward)

        # MODEL
        rtm = np.array(models[m].reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
        state = convertStimulus(np.array(models[m].state)).reshape(nb_repeat, nb_blocs, nb_trials)
        action = np.array(models[m].action).reshape(nb_repeat, nb_blocs, nb_trials)
        responses = np.array(models[m].responses).reshape(nb_repeat, nb_blocs, nb_trials)
        tmp = np.zeros((nb_repeat, 15))
        for i in xrange(nb_repeat):
            rtm[i] = center(rtm[i])
            step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i])
            tmp[i] = computeMeanRepresentativeSteps(step)[0]        
        pcrm['s'].append(state.reshape(nb_repeat*nb_blocs, nb_trials))
        pcrm['a'].append(action.reshape(nb_repeat*nb_blocs, nb_trials))
        pcrm['r'].append(responses.reshape(nb_repeat*nb_blocs, nb_trials))
        pcrm['t'].append(tmp)        

        hrtm.append(np.mean(tmp,0))
        hrt.append(data['mean'][0])
        a1 = tmp
        sys.exit()
        # rt = np.array([human.subject['fmri'][s][i]['rt'][0:nb_trials,0] for i in range(1,nb_blocs+1)])
        # rt_all.append(rt.flatten())    
        # rtm_all.append(rtm[0:nb_blocs].flatten())
        # rt = np.tile(rt, (nb_repeat, 1))        
        # CENTER
        
        
      
        # rt = center(rt)
        # action = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,1] for i in range(1,nb_blocs+1)])
        # responses = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,2] for i in range(1,nb_blocs+1)])
        # action = np.tile(action, (nb_repeat, 1))
        # responses = np.tile(responses, (nb_repeat, 1))
        # step, indice2 = getRepresentativeSteps(rt, state, action, responses)        

        super_rt[o]['model'].append(m)

    rt_all = np.array(rt_all)
    rtm_all = np.array(rtm_all)    
    pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

    for i in pcrm.iterkeys():
        pcrm[i] = np.array(pcrm[i])                
    pcrm['s'] = pcrm['s'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['a'] = pcrm['a'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['r'] = pcrm['r'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['t'] = pcrm['t'].reshape(len(p_test[o].keys())*nb_repeat, 15)
    pcr_model = extractStimulusPresentation(pcrm['r'], pcrm['s'], pcrm['a'], pcrm['r'])

    rt = (np.mean(pcrm['t'],0), sem(pcrm['t'],0))

    ht = np.reshape(human.reaction['fmri'], (14, 4*39))
    for i in xrange(len(ht)):
        ht[i] = center(ht[i])
    ht = ht.reshape(14*4, 39)    
    step, indice = getRepresentativeSteps(ht, human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
    rt_fmri = computeMeanRepresentativeSteps(step) 

    #SAVING DATA
    # data = dict()
    # data['pcr'] = dict({'model':pcr,'fmri':pcr_human})
    # data['rt'] = dict({'model':rt,'fmri':rt_fmri})
    # data['s'] = dict()
    # for i, s in zip(xrange(14), p_test[o].keys()):
    #     data['s'][s] = dict()
    #     data['s'][s]['m'] = hrtm[i]
    #     data['s'][s]['h'] = hrt[i]



    fig = figure(figsize = (15, 12))
    colors = ['blue', 'red', 'green']
    ax1 = fig.add_subplot(4,4,1)
    for i in xrange(3):
        plot(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
        errorbar(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], pcr_model['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
        plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    


    ax1 = fig.add_subplot(4,4,2)
    ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, color = 'grey', alpha = 0.5)
    ax1.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2, color = 'black', alpha = 0.9)


    for i, s in zip(xrange(14), p_test[o].keys()):
      ax1 = fig.add_subplot(4,4,i+3)
      ax1.plot(hrt[i], 'o-')
      #ax2 = ax1.twinx()
      ax1.plot(hrtm[i], 'o--', color = 'green')
      ax1.set_title(s+" "+p_test[o][s].keys()[0])

    show()

    super_data[o] = data
    hrt = np.array(hrt)
    hrtm = np.array(hrtm)
    super_rt[o]['rt'] = np.zeros((hrtm.shape[0], hrtm.shape[1], 2))
    super_rt[o]['rt'][:,:,0] = hrt
    super_rt[o]['rt'][:,:,1] = hrtm

# saving = input("Save ? : y/n : ")
# if saving == "y":
#     with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_model.pickle") , 'wb') as handle:    
#          pickle.dump(super_data, handle)
#     with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rts_all_subjects.pickle") , 'wb') as handle:    
#          pickle.dump(super_rt, handle)
