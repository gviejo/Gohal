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
from scipy.stats import norm
p_order = ['alpha','beta', 'gamma', 'noise','length','gain','sigma_bwm', 'sigma_ql']

#p = map(float, "0.784762 0.245554 0.815565 0 1 1 0.610446 0.170918".split(" "))
p = map(float, "0.40059 0.884574 0.0324776 0.0412776 0 0.0670414 0.563028 0.46275".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs

figure()
subplot(311)
plot(opt.rt_model)
plot(opt.rt)

subplot(312)
m = opt.rt_model[0]
h = opt.rt[15]
x = np.arange(-5, 10, 0.01)
# def f(x, u, v):
#     return (1/np.sqrt(2*pi*v))*np.exp(-0.5*np.power((x-u)/v, 2))
# [plot(x, f(x, i, model.parameters['sigma_ql'])) for i in [m[0]]]
# [plot(x, f(x, i, model.parameters['sigma_bwm'])) for i in m[1:]]
c, n = np.histogram(opt.rt, 100)
c = c.astype('float')
c = c/c.sum()
n = n[1:]-((n[1]-n[0])/2)
plot(n, c)
#axvline(h)


edges = opt.edges
size = edges[1]-edges[0]
[plot(edges, np.array([norm.cdf(i, j, model.parameters['sigma_ql'])-norm.cdf(i-size, j, model.parameters['sigma_ql']) for i in edges]), 'o-') for j in m]

show()

