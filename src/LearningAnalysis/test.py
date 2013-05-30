#!/usr/bin/python
# encoding: utf-8

from LearningAnalysis import SSLearning
import numpy as np
from pylab import *

responses=[0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

ssl = SSLearning(len(responses), 0.2)

ssl.EM(responses, 2000)

r = (np.random.rand(41)>0.5)*1

ssl2 = SSLearning(len(r), 0.5)

ssl2.EM(r, 2000)

plot(ssl.pmode)
ylim(0,1)

plot(ssl2.pmode)
ylim(0,1)
show()
