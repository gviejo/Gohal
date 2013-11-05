#!/usr/bin/python
# encoding: utf-8
"""
test for multiprocessing


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
from optparse import OptionParser
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import *
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from Sweep import Likelihood
from time import time

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-m", "--model", action="store", help="The name of the model to optimize", default=False)
parser.add_option("-o", "--output", action="store", help="Output directory", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------


# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel(arg):    
    for i in xrange(arg):
        #sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
    return np.array(bww.state)
# ------------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
X = human.subject['meg']
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0001
length_memory = 15
#threshold = 1.2
threshold = 1.8
#########################
#optimization parameters
fname = 'minimize'
n_run = 1000
maxiter = 1000
maxfun = 1000
xtol = 0.001
ftol = 0.001
disp = False
#########################
nb_trials = 42
#nb_blocs = 400
#########################
cats = CATS(nb_trials)
#bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, threshold)
bww = QLearning("test", cats.states, cats.actions, 0.5, 0.5, 1.0)
opt = Likelihood(human, bww, fname, n_run, maxiter, maxfun, xtol, ftol, disp)
#########################
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
t1 = time()

opt.run()
data = opt.save(options.output+"data_"+options.model+"_"+str(datetime.datetime.now()).replace(" ", "_"))

t2 = time()



print "\n"
print t2-t1
# -----------------------------------


