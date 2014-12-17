#!/usr/bin/env python
# -*- coding: utf-8 -*-
                                                                 
import sys                                                                          

sys.path.append("../../src")                                            

from Sferes import EA
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *

p_order = ['alpha', 'beta','sigma']

tmp = [0.779084, 0.0248432, 0]
model1 = QLearning2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = {}
for p,i in zip(p_order, tmp):
    parameters[p] = model1.bounds[p][0]+i*(model1.bounds[p][1]-model1.bounds[p][0])



model = QLearning(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)

with open("fmri/S5.pickle", "rb") as f:
   data = pickle.load(f)

opt = EA(data, 'S9', model)

print model.parameters

llh, lrs = opt.getFitness()

print llh, lrs


