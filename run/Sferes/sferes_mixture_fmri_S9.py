#!/usr/bin/env python
# -*- coding: utf-8 -*-
                                                               
import sys                                                                          

sys.path.append("../../src")                                            
from HumanLearning import HLearning                                                 
from Sferes import EA, RBM
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *




parameters = {'alpha': 0.61865101,
  'beta': 3.0690819999999999,
  'weight': 0.5,
  'gamma': 0.45588070000000003,
  'length': 9.2744959999999992,
  'noise': 0.11032897600000001,
  'sigma': 0.52055982999999995,
  'gain':0.5,
  'threshold': 1.0}


with open("fmri/S9.pickle", "rb") as f:
   data = pickle.load(f)

model = CSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)


opt = EA(data, 'S9', model)



llh, lrs = opt.getFitness()
print llh, lrs

sys.exit()
figure(2)
plot(opt.mean[0], 'o-')
plot(opt.mean[1], 'o--')



show()
sys.exit()






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
