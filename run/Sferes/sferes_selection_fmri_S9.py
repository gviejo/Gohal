#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python                                                       
# encoding: utf-8                                                                   
import sys                                                                          

sys.path.append("../../src")                                            
from HumanLearning import HLearning                                                 
from Sferes import EA
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *

p_order = ['beta','eta','length','threshold','noise','sigma', 'sigma_rt']

p = np.ones(7)*0.1
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

#human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
with open("fmri/S11.pickle", "rb") as f:
   data = pickle.load(f)
model = KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p] is not None:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model = KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes=True)

opt = EA(data, 'S11', model)

llh, lrs = opt.getFitness()


print llh, lrs
