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
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")

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

    def createPlot(self, tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)        
        self.plotTree(tree, (0.0, 1.0))
        plt.show()

    def plotTree(self, tree, xlimit)
        nnodes = 0
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                nnodes += 1
        step = xlimit[1]/float(nnodes)
        for k in tree.iterkeys():
            if type(tree[k]).__name__=='dict':
                            



    def plotTree2(self, myTree, parentPt, nodeTxt):
        numLeafs = self.getNumberLeafs(myTree)
        self.getTreeDepth(myTree)
        firstStr = myTree.keys()[0]
        cntrPt = (self.xOff + (1.0 + float(numLeafs))/2.0/self.totalW,\
                  self.yOff)
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, self.decisionNode)
        secondDict = myTree[firstStr]
        self.yOff = self.yOff - 1.0/self.totalD
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                self.plotTree(secondDict[key],cntrPt,str(key))
            else:
                self.xOff = self.xOff + 1.0/self.totalW
                self.plotNode(secondDict[key], (self.xOff, self.yOff),cntrPt, self.leafNode)
                self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0/self.totalD
            
    
                                                        
        
    def retrieveTree(i):
        listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
                                                     {0: 'no', 1: 'yes'}}}},
                      {'no surfacing': {0: 'no', 1: {'flippers': \
                                                     {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                      ]
        return listOfTrees[i]


