#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

test parameters and save values to the specified location

run parameterTest.py -m model -i subjectParametersmodel.txt

ex :
    
python saveValue.py -i ../../../Dropbox/ISIR/Brovelli/SubjectParameters/subjectParametersKQL.txt 
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
def loadParameters():
    p = dict()
    f = open(options.input, 'r')
    for i in f.xreadlines():
        if i[0] != "#":
            s = i.split(" ")[0]
            p[s] = dict()
            line = i.split(" ")[1].replace("(", "").replace(")\n", "").split(",")
            p[s]['p'] = dict()
            for j in line:
                if "likelihood" in j.split(":")[0]:
                    p[s][j.split(":")[0]] = float(j.split(":")[1])
                else:
                    p[s]['p'][j.split(":")[0]] = float(j.split(":")[1])
    return p

def searchStimOrder(sar):    
    # search for order
    incorrect = dict()
    for j in [1,2,3]:
        if len(np.where((sar[:,2] == 1) & (sar[:,0] == j))[0]):
            correct = np.where((sar[:,2] == 1) & (sar[:,0] == j))[0][0]
            t = len(np.where((sar[0:correct,2] == 0) & (sar[0:correct,0] == j))[0])
            incorrect[t] = j    
    if 1 in incorrect.keys() and 3 in incorrect.keys() and 4 in incorrect.keys():
        first = incorrect[1]
        second = incorrect[3]
        third = incorrect[4]
    elif len(incorrect.keys()) == 3:
        first = incorrect[incorrect.keys()[0]]
        second = incorrect[incorrect.keys()[1]]
        third = incorrect[incorrect.keys()[2]]
    elif len(incorrect.keys()) == 2:
        first = incorrect[incorrect.keys()[0]]
        second = incorrect[incorrect.keys()[1]]
        third = (set([1,2,3]) - set([first, second])).pop()
    else:
        print "You are screwed"
    return [first, second, third]

def testModel(subject):  
    model.initializeList()  
    for bloc in X[subject].iterkeys():        
        model.initialize()
        for trial in X[subject][bloc]['sar']:
            state = cvt[trial[0]]
            true_action = trial[1]-1
            values = model.computeValue(state)
            model.current_action = true_action       
            model.updateValue(trial[2])
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
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
noise = 0.01
length_memory = 8
threshold = 1.2

nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 6
cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm':BayesianWorkingMemory('bwm', cats.states, cats.actions, length_memory, noise, threshold)})

# types = dict({'kalman':np.zeros(nb_blocs, dtype = [('p_a', 'O')]), 
#               'bwm':np.zeros(nb_blocs, dtype = [('p_a', 'O')])})
types = dict({'kalman':[('p_a', 'O')],
            'bwm':['p_a', 'O']})
fields = np.array(types[options.model])[:,0]
model = models[options.model]


# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
p = loadParameters()
# -----------------------------------

# -----------------------------------
# Subject testing and saving
# -----------------------------------
cvt = dict({i:'s'+str(i) for i in [1,2,3]})
X = human.subject['meg']

for i in X.iterkeys():
    
    filename = options.output+i+"/"+options.model+".mat"
    parameter = p[i]['p']
    for j in parameter.iterkeys():
        model.setParameter(j, parameter[j])    
    testModel(i)
    x = np.zeros(len(model.value)+2, dtype = types[options.model])

    for j in xrange(len(model.value)):
        print i , j
        for k in fields:
            if k == 'p_a':
                tmp = np.matrix(model.value[j])
                order2 = searchStimOrder(X[i][j+1]['sar'])
                tmp = np.array([np.matrix(tmp[X[i][j+1]['sar'][:,0] == s]) for s in order2])
                x[j+1][k] = tmp                
    scipy.io.savemat(filename, {options.model:x})
    
# -----------------------------------
# order data
# -----------------------------------

# -----------------------------------



