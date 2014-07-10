#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8

import sys
import numpy as np
sys.path.append("../../src")

from matplotlib import *
from pylab import *

from Sferes import RBM
from sklearn.neural_network import BernoulliRBM


def genData():
    c1 = 0.5
    r1 = 0.4
    r2 = 0.3
    # generate enough data to filter
    N = 20*500
    X = array(random_sample(N))
    Y = array(random_sample(N))
    X1 = X[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
    Y1 = Y[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
    X2 = X1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
    Y2 = Y1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
    X3 = X2[ abs(X2-Y2)>0.05 ]
    Y3 = Y2[ abs(X2-Y2)>0.05 ]
    #X3 = X2[ X2-Y2>0.15 ]
    #Y3 = Y2[ X2-Y2>0.15]
    X4=zeros( 500, dtype=float32)
    Y4=zeros( 500, dtype=float32)
    for i in xrange(500):
        if (X3[i]-Y3[i]) >0.05:
            X4[i] = X3[i] + 0.08
            Y4[i] = Y3[i] + 0.18
        else:
            X4[i] = X3[i] - 0.08
            Y4[i] = Y3[i] - 0.18
    print "X", size(X3[0:500]), "Y", size(Y3)
    return(vstack((X4[0:500],Y4[0:500])))


X = genData().T
X_test = np.random.random_sample((1000, 2))

#model = BernoulliRBM(n_components=2, verbose = True)
#model.fit(X)

rbm = RBM(np.vstack(X[:,0]), np.vstack(X[:,1]), nh = 8, nbiter = 1000)

rbm.train()

Y = rbm.reconstruct(X_test, n = 10)

I = rbm.getInputfromOutput(np.vstack(X[:,1]))

subplot(311)
plot(X[:,0], X[:,1], 'o')

plot(X_test[:,0], X_test[:,1], '+')

plot(Y[:,0], Y[:,1], 'o')

subplot(312)

plot(X[:,0], X[:,1], 'o')

plot(I.flatten(), X[:,1], 'o')

subplot(313)

plot(rbm.Error)


show()
