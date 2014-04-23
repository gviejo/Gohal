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
def sample(bounds):
    tmp = dict()
    for k,b in bounds.iteritems():
        tmp[k] = np.random.uniform(low = b[0], high = b[1])
    return tmp

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
nb_repets = 10
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


def test():    
    p_test = sample(models[m].bounds)
    models[m].setAllParameters(p_test)
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
    models[m].reaction = np.array(models[m].reaction)
    
    rt = np.array([human.subject['fmri'][s][i]['rt'][0:nb_trials,0] for i in range(1,nb_blocs+1)]).flatten()
    rt = np.tile(rt, nb_repets)
    rtm = models[m].reaction.flatten()
    state = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,0] for i in range(1,nb_blocs+1)])
    action = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,1] for i in range(1,nb_blocs+1)])
    responses = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,2] for i in range(1,nb_blocs+1)])
    state = np.tile(state, (nb_repets, 1))
    action = np.tile(action, (nb_repets, 1))
    responses = np.tile(responses, (nb_repets, 1))
        
    rt = rt.reshape(nb_blocs*nb_repets, nb_trials)
    step, indice = getRepresentativeSteps(rt, state, action, responses)
    hrt = computeMeanRepresentativeSteps(step)[0]
    rtm = rtm.reshape(nb_blocs*nb_repets, nb_trials)
    step, indice = getRepresentativeSteps(rtm, state, action, responses)
    hrtm = computeMeanRepresentativeSteps(step)[0]

    hrt = np.array(hrt)
    hrtm = np.array(hrtm)
    # mi = np.array(mi)
    # pmi = np.array(pmi)

    hrt = center(hrt)
    hrtm = center(hrtm)
    diff = np.sum(np.power(hrt-hrtm, 2))
    return diff, p_test

best = None
error = 1000.
s = 'S13'
m = 'fusion'

for i in xrange(10000):
    diff, p = test()
    if diff < error:
        error = diff
        best = p
    print i, error

