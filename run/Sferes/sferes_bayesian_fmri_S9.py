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

p_order = ['length', 'noise', 'threshold']

p = map(float, "0.5 0.5 0.5 0.5".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs

ion()
plot(opt.rt_model)
plot(opt.rt)
show()