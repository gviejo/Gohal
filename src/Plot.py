#!/usr/bin/python
# encoding: utf-8
"""
Plot.py

Use matplotlib
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
import numpy as np
from fonctions import *
from matplotlib import pyplot as plt
from matplotlib import animation

class PlotCATS():
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
        

class PlotTree():
    """Class to plot trees ."""

    def __init__(self, tree, action_dict):
        self.action = action_dict
        self.decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        #self.fig = plt.figure(1,figsize=(6*3.13,4*3.13), facecolor='white')
        self.fig = plt.figure(1, figsize=(4*3.13,2*3.13), facecolor='white')
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)        
        #self.depth = self.getTreeDepth(tree)
        self.depth = 6.0
        self.depth_step = 1.0/self.depth
        self.updateTree(tree, ('',''))
        plt.show(block = False)

    def updateTree(self, tree, (state, action)):
        self.ax1.clear()
        self.plotNode(state+" "+action, (0.0, 1.0), (0.0, 1.0), self.decisionNode)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.xlimit = dict()
        step = 1/3.0
        tmp = np.array([0.0, step])
        for k in tree.iterkeys():
            cntrPt = ((tmp[1]-tmp[0])/2.0+tmp[0], 1.0-self.depth_step)            
            self.plotNode(str(k), cntrPt, (0.5,1.0), self.leafNode)
            self.xlimit[k] = tmp
            self.plotTree(tree[k], cntrPt, 1.0-self.depth_step, k)
            tmp += step
        self.fig.canvas.draw()
	#plt.pause(0.5)

    def getNumberLeafs(self, tree):
        numLeafs = 0
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                if len(tree[k].keys()) <> 0:
                    numLeafs += self.getNumberLeafs(tree[k])
                else:
                    numLeafs += 1                                             
        return numLeafs

    def getTreeDepth(self, tree):
        depth = []
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                if len(tree[k].keys()) <> 0:
                    depth.append(1+self.getTreeDepth(tree[k]))
                else:
                    depth.append(1)
        return np.max(depth)
                    
    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt,
                          xy=parentPt,
                          xycoords='axes fraction',
                          xytext=centerPt,
                          textcoords='axes fraction',
                          va="center",
                          ha="center",
                          bbox=nodeType,
                          arrowprops=self.arrow_args)

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString)
        
    def plotTree(self, tree, parent, ylevel, state):
        nnodes = 0
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                nnodes += 1
        xlimit = self.xlimit[state]
        step = (xlimit[1]-xlimit[0])/float(nnodes)
        xlimit = np.array([xlimit[0], xlimit[0]+step])
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                cntrPt = ((xlimit[1]-xlimit[0])/2.0+xlimit[0], ylevel-self.depth_step)
                self.plotNode(self.action[k], cntrPt, parent, self.leafNode)
                if 0 in tree.keys():
                    ind = tree.keys()[1:].index(k)
                    self.plotMidText(cntrPt, parent, '%.1f' % tree[0][ind])
                if len(tree[k].keys()) <> 0:
                    self.plotTree(tree[k], cntrPt, ylevel-self.depth_step, state)
                xlimit += step

###############





    def createPlot3(self, tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)        
        self.depth = self.getTreeDepth(tree)
        self.depth_step = 1.0/self.depth
        self.limit = np.array([[0.0, 1/3.0], [1/3.0, 2/3.0], [2/3.0, 1.0]]) #TODO
        self.plotTree(tree, np.array([0.0, 1.0]), 1.0)
        plt.show()

    def plotTree3(self, tree, xlimit, ylevel):
        parent = ((xlimit[1]-xlimit[0])/2.0+xlimit[0], ylevel)
        nnodes = 0
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                nnodes += 1
        step = (xlimit[1]-xlimit[0])/float(nnodes)
        xlimit = np.array([xlimit[0], xlimit[0]+step])
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                cntrPt = ((xlimit[1]-xlimit[0])/2.0+xlimit[0], ylevel-self.depth_step)
                self.plotNode(str(k), cntrPt, parent, self.leafNode)
                if len(tree[k].keys()) <> 0:
                    self.plotTree(tree[k], xlimit, ylevel-self.depth_step)
                xlimit += step
