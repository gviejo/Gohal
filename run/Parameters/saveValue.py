#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

test parameters and save values to the specified location

run parameterTest.py -m model -i subjectParametersmodel.txt

ex :
    
python saveValue.py -i kalman.txt 
                    -m kalman 
                    -o ../../../Dropbox/PEPS_GoHaL/Beh_Model/

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
import scipy.io

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-o", "--output", action="store", help="The output directory \n Must contains one subdirectory for each subject", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testParameters(subject):
   model.initializeList()
   for bloc in X[subject].iterkeys():
      sys.stdout.write("\r Sujet : %s | Blocs : %i" % (subject,bloc)); sys.stdout.flush()                    
      cats.reinitialize()
      model.initialize()
      for trial in X[subject][bloc]['sar']:
         state = cvt[trial[0]]
         true_action = trial[1]-1
         values = model.computeValue(state)
         model.current_action = true_action       
         model.updateValue(trial[2])
   model.state = convertStimulus(np.array(model.state))
   model.action = convertAction(np.array(model.action))
   model.responses = np.array(model.responses)
   model.reaction = np.array(model.reaction)


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.630     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.6666
alpha = 0.5
noise = 0.01
length_memory = 8
threshold = 1.2

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]

cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, gamma, alpha, beta)
               })
model = models[options.model]

# ----------------------------------
# Special array for matlab saving
# ----------------------------------
# types = dict({'kalman':np.zeros(nb_blocs, dtype = [('p_a', 'O')]), 
#               'bwm':np.zeros(nb_blocs, dtype = [('p_a', 'O')])})
types = dict({'kalman':[('p_a', 'O')],
              'bwm_v1':[('p_a', 'O')],
              'bwm_v2':[('p_a', 'O')],
              'qlearning':[('p_a','O')]})
#types = dict({i:[('p_a','0')] for i in models.iterkeys()})
fields = np.array(types[options.model])[:,0]


# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
p = eval(open(options.input, 'r').read())

# -----------------------------------

# -----------------------------------
# Subject testing and saving
# -----------------------------------
cvt = dict({i:'s'+str(i) for i in [1,2,3]})
X = human.subject['meg']

for i in p.iterkeys():    
    filename = options.output+i+"/"+options.model+".mat"
    for j in p[i].iterkeys():
       model.setParameter(j, p[i][j])
    
    testParameters(i)
    x = np.zeros(len(model.value)+2, dtype = types[options.model])
    
    for j in xrange(len(model.value)):
        for k in fields:
            if k == 'p_a':
                tmp = np.matrix(model.value[j])
                
                order2 = searchStimOrder(X[i][j+1]['sar'][:,0], X[i][j+1]['sar'][:,1], X[i][j+1]['sar'][:,2])
                
                x[j+1][k] = np.array([np.matrix(tmp[X[i][j+1]['sar'][:,0] == s]) for s in order2])  
    scipy.io.savemat(filename, {options.model:x})

    
# -----------------------------------
# order data
# -----------------------------------

# -----------------------------------



