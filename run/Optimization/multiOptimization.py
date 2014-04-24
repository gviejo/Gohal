#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test for multiprocessing


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser
sys.path.append("../../src")
#sys.path.append("/home/viejo/Gohal/src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import *
from Selection import *
if os.uname()[1] in ['atlantis', 'paradise']:
	from matplotlib import *
	from pylab import *
from HumanLearning import HLearning
from Sweep import *


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
parser.add_option("-f", "--fonction", action="store", help="Scipy function", default=False)
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
#human = HLearning(dict({'meg':('/home/viejo/Beh/MEG/Beh_Model/',48), 'fmri':('/home/viejo/Beh/fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
#########################
#optimization parameters
n_run = 5000
n_grid = 10
maxiter = 5000
maxfun = 10000
xtol = 0.01
ftol = 0.01
disp = False
#########################

cats = CATS(0)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions)})

#opt = Likelihood(human, models[options.model], options.fonction, n_run, n_grid, maxiter, maxfun, xtol, ftol, disp)
opt = SamplingPareto(human.subject['fmri'], models[options.model], maxiter)
#########################
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------


opt.run()
opt.save(options.output)


# -----------------------------------


