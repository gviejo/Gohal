#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

test parameters and save values to the specified location

run parameterTest.py -m model -i subjectParametersmodel.txt

ex :
    
python saveValue.py -i kalman.txt 
                    -m kalman 
                    -o ../../../Dropbox/PEPS_GoHaL/MEG/

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
parser.add_option("-i", "--input", action="store", help="The name of the parameters to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-o", "--output", action="store", help="The output directory \n Must contains one subdirectory for each subject", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testParameters(subject):
   model.initializeList()
   s = subject
   #Dirty Trick to have ordered data
   if list(subject)[1] == '0':
        s = "".join([list(subject)[0], list(subject)[2]])
   for bloc in X[s].iterkeys():
      sys.stdout.write("\r Sujet : %s | Blocs : %i" % (s,bloc)); sys.stdout.flush()                    
      cats.reinitialize()
      model.initialize()
      for trial in X[s][bloc]['sar']:
         state = cvt[trial[0]]
         true_action = trial[1]-1         
         values = model.computeValue(state)         
         model.current_action = true_action       
         model.updateValue(trial[2])
   model.state = convertStimulus(np.array(model.state))
   model.action = convertAction(np.array(model.action))
   model.responses = np.array(model.responses)
   


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
types = dict({'kalman':[('p_a', 'O'),
                        ('entropy','O'),
                        ('vpi','O')],
              'qlearning':[('p_a','O'),
                          ('entropy','O')],
              'bwm_v1':[('p_a', 'O'),
                        ('entropy','O'),
                        ('p_r_s', 'O'),
                        ('nb_inf','O')],                                            
              'bwm_v2':[('p_a', 'O'),
                        ('entropy','O'),
                        ('p_r_s', 'O'),
                        ('nb_inf','O')]                        
              })
#types = dict({i:[('p_a','0')] for i in models.iterkeys()})
fields = np.array(types[options.model])[:,0]


# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
p = eval(open(options.input, 'r').read())[options.model]

# -----------------------------------

# -----------------------------------
# Subject testing and saving
# -----------------------------------
cvt = dict({i:'s'+str(i) for i in [1,2,3]})
X = human.subject['meg']

filename = options.output+options.model+".mat"
x = np.zeros((len(X.keys()), 4), dtype = types[options.model])

for i in sorted(p.keys()):    
    for j in p[i].iterkeys():
       model.setParameter(j, p[i][j])    
    testParameters(i)    
    for j in xrange(len(model.value)):        
        for k in fields:
            if k == 'p_a':                
                x[sorted(p.keys()).index(i),j][k] = np.matrix(model.value[j])
            if ('kalman' or 'qlearning') in options.model:
                if k == 'entropy':
                    x[sorted(p.keys()).index(i),j][k] = np.matrix(np.vstack(model.reaction[j]))    
                if options.model == 'kalman' and k == 'vpi':
                    x[sorted(p.keys()).index(i),j][k] = np.matrix(np.vstack(model.vpi[j]))    
            elif 'bwm' in options.model:                    
                if k == 'entropy':
                    x[sorted(p.keys()).index(i),j][k] = np.matrix(np.vstack(model.entropies[j]))
                elif k == 'p_r_s':
                    x[sorted(p.keys()).index(i),j][k] = np.matrix(np.vstack(model.sample_p_r_s[j]))    
                elif k == 'nb_inf':
                    x[sorted(p.keys()).index(i),j][k] = np.matrix(np.vstack(model.sample_nb_inf[j]))    
scipy.io.savemat(filename, {options.model:x})

    




