#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and test parameters

run parameterTest.py -i ../../../Dropbox/ISIR/Plot/optparameters.py

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
parser.add_option("-s", "--subject", action="store", help="Which subject to plot \n Ex : -s S1", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def iterationStep(iteration, models, display = True):
    state = cats.getStimulus(iteration)
    for m in models.itervalues():
        action = m.chooseAction(state)
        reward = cats.getOutcome(state, action)
        m.updateValue(reward)

def omegaFunc(cible, freq1, freq2):
    if cible == freq1 == freq2 == 0:
        return 0.0
    elif freq1 == freq2:
        w = float(cible/freq1)
        if w < 0.0:
            return 0.0
        elif w > 1.0:
            return 1.0
        else:
            return w        
    else:
        w = float((cible-freq2)/(freq1-freq2))
        if w < 0.0:
            return 0.0
        elif w > 1.0:
            return 1.0
        else:
            return w

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
noise_width = 0.01
correlation = "diff"
length_memory = 15

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 46

cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, alpha, beta, gamma)
              })

# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
#p = dict()
#for m in models.iterkeys():    
#    f = open(m, 'rb')
#    p[m] = pickle.load(f)
X = np.load(options.input)
sys.exit()
est = KMeans(5)
est.fit(X)

X = est.cluster_centers_


omega = []
order = ['beta', 'gamma', 'length', 'noise']

# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------

opt = Optimization(human, cats, nb_trials, nb_blocs)

data = dict()
data['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

for p,t in zip(X,xrange(len(X))):
    print str(t)+" | "+str(len(X))
    models['kalman'].beta = p[0]
    models['kalman'].gamma = p[1]
    models['bmw'].length_memory = p[2]
    models['bmw'].noise_width = p[3]
    for m in models.iterkeys():    
        opt.testModel(models[m])
        models[m].state = convertStimulus(np.array(models[m].state))
        models[m].action = convertAction(np.array(models[m].action))
        models[m].responses = np.array(models[m].responses)
        data[m] = extractStimulusPresentation(models[m].responses, models[m].state, models[m].action, models[m].responses)
    tmp = np.zeros((3,10))
    for i in xrange(3):
        for j in xrange(10):
            tmp[i,j] = omegaFunc(data['meg']['mean'][i,j],data['bmw']['mean'][i,j],data['kalman']['mean'][i,j])
    omega.append(tmp)


omega = np.array(omega)

# -----------------------------------
# Plot
# -----------------------------------

params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':11,
          'ytick.labelsize':11,
          'text.usetex':False}
dashes = ['-', '--', ':']


fig = figure(figsize=(10, 5))
rcParams.update(params)                  
m = np.mean(omega, axis = 0)
v = np.var(omega, axis = 0)
for i,l in zip([1,2,3],xrange(3)):
    subplot(2,3,i)
    plot(range(1, len(m[l])+1), m[l], linewidth = 2, color = 'black')
    fill_between(range(1, len(m[l])+1), m[l]-v[l],m[l]+v[l], alpha = 0.4, color = 'grey')
    ylabel("$\omega$", fontsize = 13)
    xlabel('Trial')
    yticks(np.arange(0, 1.2, 0.2))
    xticks(range(2,11,2))
    xlim(0.8, 11)
    title("Stimulus "+str(i), fontsize = 18)
    grid()

for i,l in zip([4,5,6],xrange(3)):
    subplot(2,3,i)
    bar(range(1,11), v[l], color = 'black')
    xlim(0.8, 11)
    ylim(0, np.max(v)+0.05)
    ylabel("$\sigma^2$", fontsize = 13)
    grid()
    xlabel('Trial')

figtext(0.04, 0.94, 'A', fontsize = 20)
figtext(0.04, 0.47, 'B', fontsize = 20)


subplots_adjust(left = 0.08, wspace = 0.4, right = 0.86, hspace = 0.35)

#fig.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig6.pdf', bbox_inches='tight')
show()
