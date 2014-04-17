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

from scipy.optimize import curve_fit, leastsq




model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])

p_order = ['alpha','beta', 'gamma', 'noise','length','gain','threshold', 'sigma']

p = map(float, "1 0.130219 0.1 0.420989 0.0154722 0 1 0.998014".split(" "))

tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])

# parameters = dict({'alpha': 0.92961300000000002,
# 				 'beta': 1.1455569999999999,
# 				 'gain': 1.0000000000000001e-05,
# 				 'gamma': 0.78684500000000002,
# 				 'length': 9.3192839999999997,
# 				 'noise': 0.010792700000000001,
# 				 'threshold': 9.3932982999999997})

# parameters = dict({'alpha': 0.92961300000000002,
# 				 'beta': 2.1455569999999999,
# 				 'gain': 1.0,
# 				 'gamma': 0.78684500000000002,
# 				 'length': 8,
# 				 'noise': 0.010792700000000001,
# 				 'threshold': 1.0,
# 				 'sigma':0.8})

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters)


opt = EA(human.subject['fmri']['S9'], 'S9', model)



llh, lrs = opt.getFitness()
print llh, lrs

figure(2)
plot(opt.fitfunc(opt.pa[0], opt.mean[1][0]), '+-')
plot(opt.mean[0][0], 'o-')







left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]
plt.figure(1, figsize=(12,9))
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
axScatter.imshow(opt.p, interpolation = 'nearest', origin = 'lower')
axHistx.plot(opt.p_rtm)
axHisty.plot(opt.mass, np.arange(len(opt.mass)))
#axHistx.set_title("dh/n")
#axHistx.set_title("dh")
axHistx.set_title("n")
plt.text(s = "MI "+str(lrs), x = 0.0, y = 16.0)


plt.figure(3, figsize=(8,8))
rt = opt.rt
ind = np.argsort(rt)
subplot(211)
plot(rt[ind])
subplot(212)
plot(opt.rtm[ind])


show()
sys.exit()

#np.save("rtm_h_n", opt.rtm[ind])
#np.save("rtm_h", opt.rtm[ind])
np.save("rtm_n", opt.rtm[ind])


hn = np.load("rtm_h_n.npy")
h = np.load("rtm_h.npy")
n = np.load("rtm_n.npy")

figure(figsize = (9, 14))
subplot(411)
plot(rt[ind])
subplot(412)
plot(hn)
title("h/n")
subplot(413)
plot(h)
title("h")
subplot(414)
plot(n)
title("n")
savefig("/home/viejo/Desktop/meeting_11_04_14/cumul_fusion.pdf")


sys.exit()

def createDataSet(x):
	bin_size = 2*(np.percentile(x, 75)-np.percentile(x, 25))*np.power(len(x), -(1/3.))	
	if bin_size < (x.max()-x.min())/30.:
		bin_size = (x.max()-x.min())/30.
		mass, edges = np.histogram(x, bins = np.linspace(x.min(), x.max()+bin_size, num = 31, endpoint = True))
	else:
		mass, edges = np.histogram(x, bins = np.arange(x.min(), x.max()+bin_size, bin_size))

	position = np.digitize(x, edges)-1
	
	xx = np.zeros((x.shape[0], mass.shape[0]))
	for i in xrange(len(position)): xx[i, position[i]] = 1.0
	
	return xx

x = createDataSet(np.tile(opt.rt, opt.n_repets))
y = createDataSet(opt.rtm)

rbm = RBM(x, y, nh = 10, nbiter = 800)

rbm.train()

I = rbm.getInputfromOutput(y)

#I = rbm.reconstruct(np.random.random_sample(size = (rbm.nd, rbm.nv)))

map(plot, I[:,0:rbm.nx])

show()
