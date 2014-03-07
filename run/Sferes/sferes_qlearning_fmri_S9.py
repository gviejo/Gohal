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

p_order = ['alpha', 'beta', 'gamma', 'sigma']

p = map(float, "0.66469 1 0.0360993 0".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = QLearning(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs

figure()
subplot(211)
plot(opt.rt_model)
plot(opt.rt)

subplot(212)
m = opt.rt_model[15]
h = opt.rt[15]
x = np.arange(-10, 10, 0.01)
def f(x, u, v):
    return (1/np.sqrt(2*pi*v))*np.exp(-0.5*np.power((x-u)/v, 2))
[plot(x, f(x, i, model.parameters['sigma'])) for i in [m[0]]]


axvline(h)
show()
