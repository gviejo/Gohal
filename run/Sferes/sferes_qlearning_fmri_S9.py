#!/usr/bin/env python
# -*- coding: utf-8 -*-
                                                                 
import sys                                                                          

sys.path.append("../../src")                                            
from HumanLearning import HLearning                                                 
from Sferes import EA
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *
from scipy.stats import norm

p_order = ['alpha', 'beta','sigma']


p = map(float, "0.66469 1 0.0360993 0.5".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]


parameters = tmp

model = QLearning(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)

with open("fmri/S9.pickle", "rb") as f:
   data = pickle.load(f)

opt = EA(data, 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs


