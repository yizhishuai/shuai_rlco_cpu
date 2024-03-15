# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 11:22:57 2022

@author: Mieszko Ferens
"""

import matplotlib.pyplot as plt

# Close all figures (in case some remain from a previous execution)
plt.close('all')
# 参数： x轴， 平均奖励（最近10000） 0  labels是标签 [0] x [1] y  [2] agent名称
# Funtion that creates, plots and labels a figure with given parameters
def makeFigurePlot(x_axis, y_axis, optimal=None, labels=[], legend=[],
                   log=False):
    
    # Insert optimal value to y_axis so it is show on the graph (as first
    # element so it's always the same colour)
    if(optimal != None):
        line_styles = (['-']*len(y_axis))
        line_styles.insert(0, '--')
        y_axis.insert(0, [optimal]*len(x_axis))
        if(legend != []):
            legend.insert(0, 'Optimal')
    else:
        line_styles = (['-']*len(y_axis))
    
    # Create and plot figure
    plt.figure(figsize=[10,5])
    for a in range(len(y_axis)):
        plt.plot(x_axis, y_axis[a], line_styles[a])
    
    if(log):
        plt.yscale('log')
    
    # Label figure
    if labels:
        plt.xlabel(labels[0])#, fontsize=22)
        plt.ylabel(labels[1])#, fontsize=22)
        plt.title(labels[2])#, fontsize=22)
    if legend:
        plt.legend(legend)#, fontsize=16)

# Funtion that creates, fills and labels a histogram with given parameters
def makeFigureHistSingle(y_axis, bins=10, labels=[], legend=[], thresh=None):
    
    # Create and plot figure
    plt.figure(figsize=[10,5])
    if(len(y_axis) == 1):
        plt.hist(y_axis, bins=bins)
    else:
        plt.hist(y_axis, bins=bins, alpha=0.5, histtype='step')
    
    # Add threshhold line
    if(thresh):
        plt.axvline(thresh, color='k', linestyle='--')
        legend.insert(0, 'Max tolerable delay')
    # Label figure
    if labels:
        plt.xlabel(labels[0])#, fontsize=22)
        plt.ylabel(labels[1])#, fontsize=22)
        plt.title(labels[2])#, fontsize=22)
    if legend:
        plt.legend(legend)#, fontsize=16)

# Funtion that creates, fills and labels subplot histograms with given
# parameters
def makeFigureHistSubplot(y_axis, bins=10, labels=[], legend=[], thresh=None):
    
    # Create and plot figure
    plt.figure(figsize=[10,10])
    
    for hist in range(len(y_axis)):
        plt.subplot(len(y_axis), 1, hist+1)
        plt.hist(y_axis[hist])
        # Add threshhold line #TODO (issue with bar visualization)
        #if(thresh):
            #plt.xlim([-5 , max(y_axis[hist] + [thresh]) + 10])
            #plt.axvline(thresh, color='k', linestyle='--')
            #legend.insert(0, 'Max tolerable delay')
        # Label figure
        if labels:
            plt.xlabel(labels[0])#, fontsize=22)
            plt.ylabel(labels[1])#, fontsize=22)
        if legend:
            plt.title(legend[hist])#, fontsize=16)
    if labels:
        plt.suptitle(labels[2])#, fontsize=22)
    
    plt.tight_layout()

