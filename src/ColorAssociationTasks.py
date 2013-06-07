#!/usr/bin/python
# encoding: utf-8
"""
ColorAssociationTask.py

Class that implement the visuo-motor learning task
as described in Brovelli & al, 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *

class CATS():
    """ Class that implement the visuo-motor learning task
    as described in Brovelli & al, 2011 """
    
    def __init__(self, nb_trials = 42, case = 'meg'):
        self.nb_trials = nb_trials
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.asso = self.createAssociationDict(self.states)
        self.used = []
        self.correct = []
        self.stimuli = self.createStimulusList(self.states, nb_trials)
        self.time = [1, 3, 4]
        self.incorrect = dict()
        for i in xrange(len(self.states)):
            self.incorrect[self.states[i]] = self.time[i]

    def chooseExperiment(self, case):
        if case == 'meg':            
            return [3, 9, 15]
        elif case == 'fmri':
            return [3, 9, 15]
        
    def reinitialize(self, nb_trials, case):
        self.__init__(nb_trials, case)

    def createAssociationDict(self, states):
        tmp = dict()
        for i in states:
            tmp[i] = dict()
        return tmp

    def getStimulus(self, iteration):
        try:
            return self.stimuli[iteration]
        except:
            print "Error: no more stimuli"
            sys.exit(0)
                        
    def createStimulusList(self, states, nb_trials):
        tmp = list(states)
        s = []
        for i in xrange((nb_trials/len(states))+1):
            np.random.shuffle(tmp)
            s.append(list(tmp))
        return (np.array(s)).flatten()

    def getOutcome(self, state, action):
        tmp = self.asso[state].values()
        if state in self.asso.keys() and action in self.asso[state].keys():
            return self.asso[state][action]
        elif tmp.count(-1) >= self.incorrect[state] and tmp.count(1) == 0:
            self.asso[state][action] = 1
            self.used.append(action)
            self.correct.append(state+" => "+action)
            return 1
        else:
            self.asso[state][action] = -1
            return -1
            

    '''                                            
    def getOutcome(self, state, action, i):
        if i < self.time[0]:
            self.asso[state][action] = -1
            return -1
        elif i >= self.time[0] and action in self.asso[state].keys():
            return self.asso[state][action]
        elif i >= self.time[0] and len(self.correct) == 0:
            self.asso[state][action] = 1
            self.used.append(action)
            self.correct.append(state+' => '+action)
            return 1
        elif i >= self.time[1] and len(self.correct) == 1 and action not in self.used:
            self.asso[state][action] = 1
            self.used.append(action)
            self.correct.append(state+' => '+action)
            return 1
        elif i >= self.time[2] and len(self.correct) == 2 and action not in self.used:
            self.asso[state][action] = 1
            self.used.append(action)
            self.correct.append(state+' => '+action)
            return 1
        else:
            self.asso[state][action] = -1
            return -1

    def getOutcome(self, state, action, i):
        if i < 3:
            self.asso[state][action] = -1
            return -1
        elif i >= 3 and action in self.asso[state].keys():
            return self.asso[state][action]
        else:
            if state == 's1' and 1 not in self.asso['s1'].values():
                self.asso[state][action] = 1
                self.correct.append(state+' => '+action)
                return 1
            elif state == 's2' and np.sum(self.asso['s2'].values()) <= -3:
                if np.max([self.asso[s][action] for s in self.states if action in self.asso[s].keys()]+[0]) <> 1:
                    self.asso[state][action] = 1
                    self.correct.append(state+' => '+action)
                    return 1
                else:
                    self.asso[state][action] = -1
                    return -1
            elif state == 's3' and np.sum(self.asso['s3'].values()) <= -4:
                if np.max([self.asso[s][action] for s in self.states if action in self.asso[s].keys()]+[0]) <> 1:
                    self.asso[state][action] = 1
                    self.correct.append(state+' => '+action)
                    return 1
                else:
                    self.asso[state][action] = -1
                    return -1
            else:
                self.asso[state][action] = -1
                return -1
      '''













