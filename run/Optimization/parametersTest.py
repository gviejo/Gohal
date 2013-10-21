#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and test parameters

run parameterTest.py -m model -i subjectParametersmodel.txt

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
from sklearn.cluster import KMeans

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

def testModel(nb_blocs = 4):    
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        model.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)
            model.updateValue(reward)
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
nb_blocs = 40
cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm':BayesianWorkingMemory('bwm', cats.states, cats.actions, length_memory, noise, threshold)})

model = models[options.model]

# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
p = loadParameters()
# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
model.initializeList()
for i in p.iterkeys():
    for j in p[i]['p'].iterkeys():
        model.setParameter(j, p[i]['p'][j])    
    testModel()    

model.state = convertStimulus(np.array(model.state))
model.action = convertAction(np.array(model.action))
model.responses = np.array(model.responses)


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (9,4))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          
#rcParams.update(params)                  
colors = ['blue', 'red', 'green']
subplot(1,1,1)
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

show()

#fig.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig6.pdf', bbox_inches='tight')

