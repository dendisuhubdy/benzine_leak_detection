# -*- coding: utf-8 -*-
# Simulate Sensor Data by combining QUIC model output with Brownian Baseline
# 11/30/14

import numpy as np
import numpy.random as nr
import datetime as dt
import copy as copy
import pandas as pd
import windrose as wr
from ggplot import *
import math
import scipy.signal as signal
import math
from matplotlib import pyplot as plt
import patsy
import statsmodels.formula.api as smf
import scipy.linalg as LA
import matplotlib as mpl
from SPOD_functions import *
import os
os.chdir("/Users/halleybrantley/Desktop/Dropbox/ST758_final")

nr.seed(seed=79819)
plt.close('all')
datafolder = "./data/"
figfolder = "./figures/"
SimData = "./simulated/Simulated_07052014_5min.csv"
freq = '3H'
freqT = 3
avgTime = '5min'
thresh1 = 0.08
thresh2 = 0.20
smoothingWindow = 40

QUIC20 = pd.read_csv(SimData)
startTime = pd.Timestamp(dt.datetime(2014, 07, 05, 12, 00, 20))
endTime = pd.Timestamp(dt.datetime(2014, 07, 05, 18,  00, 00))
TimeStamp = pd.date_range(startTime, endTime, freq='20s')
QUIC20['dateTime'] = pd.date_range(startTime, endTime, freq='20s')
QUIC20.BaseSim = QUIC20.BaseSim*10**7
QUIC20.RemoteSim = QUIC20.RemoteSim*10**7
QUIC20.index = QUIC20.dateTime
QUIC = QUIC20.resample('1s', fill_method = 'pad')
QUIC['dateTime'] = QUIC.index.copy()

# Add real wind measurements to simulated sensor measurements
rawdat = importSPOD(datafolder, 1, startTime, endTime)
QUIC['U'] = rawdat['U']
QUIC['V'] = rawdat['V']
QUIC['WS'] = rawdat['WS']
QUIC['Time'] =  pd.to_datetime(QUIC.index.copy()).astype('int').astype(float)/(10**18)

# Simulate Data
Num = len(QUIC)
QUIC['BaseBM'] = genBrownianBridge (Num) + 1.5

NewBase  = QUIC.BaseSim + QUIC.BaseBM
NewBase[NewBase > 5] = 5
NewBase[NewBase < 0.25] = nr.randn(len(NewBase[NewBase < 0.25]))*.05 + 0.25
QUIC['Base'] = NewBase

QUIC['RemoteBM'] = genBrownianBridge (Num)
NewRemote  = QUIC.RemoteSim + QUIC.RemoteBM
NewRemote[NewRemote > 5] = 5
NewRemote[NewRemote < 0.25] = nr.randn(len(NewRemote[NewRemote < 0.25]))*.05 + 0.25
QUIC['Remote'] = NewRemote

# Plot Simulated Data
font = {'weight' : 'bold',
        'size'   : 8}
mpl.rc('font', **font)

baseTotal = ggplot(aes(x='dateTime', y='Base'), data=QUIC) +\
    geom_line() +\
    ylim(0,5) +\
    geom_line() + xlab("") + ylab("Simulated Signal (V)")
 
baseRand = ggplot(aes(x='dateTime', y='BaseBM'), data=QUIC) +\
    geom_line() + xlab("") + ylab("Stochastic Baseline")

baseSim = ggplot(aes(x='dateTime', y='BaseSim'), data=QUIC) +\
    geom_line()+\
    ylim(0,5) + xlab("") + ylab("Simulated Signal (V)")

remoteTotal = ggplot(aes(x='dateTime', y='Remote'), data=QUIC) +\
    geom_line() + xlab("") + ylab("Simulated Signal (V)")

ggsave(plot = baseTotal, filename = figfolder+'BaseTotal.png', width = 8, height = 2)
ggsave(plot = baseRand, filename = figfolder+'BaseRand.png', width = 8, height = 2)
ggsave(plot = baseSim, filename = figfolder+'BaseSim.png', width = 8, height = 2)
ggsave(plot = remoteTotal, filename = figfolder+'remoteTotal.png', width = 8, height = 2)

# Illustrate Method
fitMinSpline(QUIC['Base'][QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')],
 QUIC['Time'][QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')], smoothingWindow, plot=True, plotVar = QUIC.dateTime)
ggsave(filename = figfolder+'Spline_fit.png', width = 8, height = 2)

QUICFilt = applyFilters(QUIC[QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')], thresh1, thresh2, smoothingWindow)
butterplot = ggplot(aes(x='dateTime', y='butterBase'), data=QUICFilt) + geom_line() +\
            ylim(0,5) +\
            xlab('') + ylab('Sensor after Butterworth')
ggsave(plot=butterplot, filename = figfolder+'Butterworth_filt.png', width = 8, height = 2)

# Apply algorithm
QUIC['TrueBase'] = QUIC.BaseSim.apply(isSignal, args = (.1,))
QUIC['TrueRemote'] = QUIC.RemoteSim.apply(isSignal, args = (0.1,))

FiltAvg = piecewiseImportSpod(startTime, endTime, freq, avgTime, thresh1, thresh2, smoothingWindow, QUIC, True)

remoteTotal = ggplot(aes(x='dateTime', y='Remote'), data=QUIC) +\
    geom_line() +\
    theme_matplotlib(mpl.rc('font', **font), matplotlib_defaults=False)

TrueVDetect = ggplot(aes(x='TrueBase', y='butterBaseSignal'), data=FiltAvg) +\
    geom_point(color = 'blue') +\
    geom_point(aes(x = 'TrueRemote', y='butterRemoteSignal'), color = 'blue') +\
    geom_point(aes(y='splineBaseSignal'), color='green') +\
    geom_point(aes(x = 'TrueRemote', y='splineRemoteSignal'), color='green') +\
    geom_abline(aes(intercept = 0, slope=1)) +\
    ylab('Detected Signal 5 min mean') +\
    xlab('True Signal 5 min mean')

ggsave(plot = TrueVDetect, filename = figfolder+'TrueVDetect.png', width = 4.5, height = 4)

ButterCorrect = (len(FiltAvg[(FiltAvg.butterBaseSignal > 0.017)&(
        FiltAvg.TrueBase > 0.017)]) + len(FiltAvg[(FiltAvg.butterRemoteSignal > 0.017)&(
        FiltAvg.TrueRemote > 0.017)]))/(2.0*len(FiltAvg))
print("Butter percent correct: " + str(ButterCorrect))

SplineCorrect = (len(FiltAvg[(FiltAvg.splineBaseSignal > 0.017)&(
        FiltAvg.TrueBase > 0.017)]) + len(FiltAvg[(FiltAvg.splineRemoteSignal > 0.017)&(
        FiltAvg.TrueRemote > 0.017)]))/(2.0*len(FiltAvg))
print("Spline percent correct: " + str(SplineCorrect))

ButterFalsePos = (len(FiltAvg[(FiltAvg.butterBaseSignal > 0.017)&(
        FiltAvg.TrueBase < 0.017)]) + len(FiltAvg[(FiltAvg.butterRemoteSignal > 0.017)&(
        FiltAvg.TrueRemote < 0.017)]))/(2.0*len(FiltAvg))
print("Butter percent false pos: " + str(ButterFalsePos))

SplineFalsePos = (len(FiltAvg[(FiltAvg.splineBaseSignal > 0.017)&(
        FiltAvg.TrueBase < 0.017)]) + len(FiltAvg[(FiltAvg.splineRemoteSignal > 0.017)&(
        FiltAvg.TrueRemote < 0.017)]))/(2.0*len(FiltAvg))
print("Spline percent false pos: " + str(SplineFalsePos))
