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
parser.add_option("-o", "--output", action="store", help="The output file with pcr and rt for each model", default=False)

(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def convertStimulus_(s):
    return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

def loadParameters():
    return eval(open(options.input, 'r').read())

def testParameters():
    data = loadParameters()
    pcr = dict()
    rt = dict()
    for m in data.iterkeys():      
        model = models[m]
        model.startExp()
        for s in data[m].iterkeys():
            model.setAllParameters(data[m][s])
            for i in xrange(nb_blocs):        
                cats.reinitialize()
                cats.stimuli = np.array(map(convertStimulus_, human.subject['fmri'][s][i+1]['sar'][0:nb_trials,0]))
                model.startBloc()
                for j in xrange(nb_trials):
                    sys.stdout.write("\r Model : "+m+" | Sujet : "+s+"| Blocs : "+str(i)+" | Trials : "+str(j));sys.stdout.flush()
                    state = cats.getStimulus(j)
                    action = model.chooseAction(state)
                    reward = cats.getOutcome(state, action)
                    model.updateValue(reward)
            tmp = np.array(model.reaction[-nb_blocs:])
            tmp = tmp-np.min(tmp)
            tmp = tmp/float(np.max(tmp))
            for i,j in zip(xrange(-nb_blocs, 0), xrange(len(tmp))):
                model.reaction[i] = list(tmp[j])
        model.state = convertStimulus(np.array(model.state))
        model.action = convertAction(np.array(model.action))
        model.responses = np.array(model.responses)
        model.reaction = np.array(model.reaction)
        pcr[m] = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
        step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
        rt[m] = computeMeanRepresentativeSteps(step)
    return pcr, rt
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))

# # -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_trials = human.responses['fmri'].shape[1]
nb_blocs = 4
cats = CATS(nb_trials)

models = dict({"fusion":FSelection(cats.states, cats.actions, {'alpha':0.0,'beta':0.0,'gamma':0.0,'length':0.0,'noise':0.0,'threshold':0.0,'gain':0.0}),
              "qlearning":QLearning(cats.states, cats.actions, {'alpha':0.0, 'beta':0.0, 'gamma':0.0}),
              "bayesian":BayesianWorkingMemory(cats.states, cats.actions, {'length':0.0, 'noise':0.0, 'threshold':0.0}),
              "keramati":KSelection(cats.states, cats.actions,{"gamma":0.0,"beta":1.0,"eta":0.0001,"length":10.0,"threshold":0.0,"noise":0.0,"sigma":0.0})})

# ------------------------------------
# Parameters testing
# ------------------------------------
t1 = time()
pcr, rt = testParameters()
t2 = time()

print "\n"
print t2-t1

# ----------------------------------
# SAving data
# ----------------------------------

if options.output:
    saveData(options.output, dict({'rt':rt,'pcr':pcr}))
  

