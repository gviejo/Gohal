#!/usr/bin/python                                                       
# encoding: utf-8                                                                   
import sys                                                                          

sys.path.append("../../src")                                            
from HumanLearning import HLearning                                                 
from Sferes import EA, RBM
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *
from scipy.stats import norm
#p_order = ['alpha','beta', 'gamma', 'noise','length','gain','sigma_bwm', 'sigma_ql']
p_order = ['alpha','beta', 'gamma', 'noise','length','gain']

#p = map(float, "0.784762 0.245554 0.815565 0 1 1 0.610446 0.170918".split(" "))
p = map(float, "0.118246 0.897181 0.946751 0.595232 0.092146 0.922554 0.126166 0.341448".split(" "))
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


xx = np.zeros((156,opt.position.max()+1))
for i in xrange(len(opt.position)):
	xx[i][opt.position[i]] = 1.0
yy = model.pdf

# xx = np.zeros((100, 2))
# xx[0:50] = np.random.beta(0.5, 5, size = (50, 2))
# xx[50:] = np.random.beta(5, 0.5, size = (50, 2))
# yy = np.zeros((100, 1))
# yy[0:50] = np.random.beta(0.5, 5, size = (50, 1))
# yy[50:] = np.random.beta(5, 0.5, size = (50, 1))


bolt = RBM(xx, yy, nb_iter = 10000)
bolt.train()
sys.exit()





figure()
subplot(311)
plot(opt.rt_model)
plot(opt.rt)

subplot(312)
m = opt.rt_model[0]

x = np.arange(-5, 10, 0.01)
# def f(x, u, v):
#     return (1/np.sqrt(2*pi*v))*np.exp(-0.5*np.power((x-u)/v, 2))
# [plot(x, f(x, i, model.parameters['sigma_ql'])) for i in [m[0]]]
# [plot(x, f(x, i, model.parameters['sigma_bwm'])) for i in m[1:]]
c, n = np.histogram(opt.rt, opt.edges)
c = c.astype('float')
c = c/c.sum()
n = n[1:]-((n[1]-n[0])/2)
plot(n, c)


[plot(opt.edges, np.array([norm.cdf(i, j, model.parameters['sigma_ql'])-norm.cdf(i-opt.bin_size, j, model.parameters['sigma_ql']) for i in opt.edges]), 'o-') for j in m]



subplot(313)
plot(opt.rt, opt.s, 'o')

show()