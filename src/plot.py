#!/usr/bin/python
# encoding: utf-8
"""
plot.pu

multi class files to plot data

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
from pylab import *

class plotCATS():
    """Class that allow to plot in different way results
    from the color association task ."""

    def __init__(self, data):
        self.data = data

    def plotPerformanceAll(self):
        tmp = []
        for i in self.data.iterkeys():
            for j in self.data[i].iterkeys():
                tmp.append(self.data[i][j]['sar'][0:42,2])
        mean = np.mean(tmp, 0)
        var = np.var(tmp, 0)
        plot(mean, label = 'human performance')
        fill_between(range(len(mean)), mean-var/2, mean+var/2, facecolor = 'green', interpolate = True, alpha = 0.1)
        legend()
        grid()
        show()

    def plotComparison(self, dic_data):
        tmp = []
        for i in self.data.iterkeys():
            for j in self.data[i].iterkeys():
                tmp.append(self.data[i][j]['sar'][0:42,2])
        mean = np.mean(tmp, 0)
        var = np.var(tmp, 0)        
        plot(mean,'o-', label = 'human performance')
        fill_between(range(len(mean)), mean-var/2, mean+var/2, facecolor = 'green', interpolate = True, alpha = 0.1)

        for i in dic_data.iterkeys():
            mean2 = np.mean(dic_data[i], 0)
            var2 = np.var(dic_data[i], 0)
            plot(mean2,'o-', label = i+' performance')
            fill_between(range(len(mean2)), mean2-var2/2, mean2+var2/2, facecolor = 'green', interpolate = True, alpha = 0.1)
        legend()
        grid()

        show()
        
        
        
