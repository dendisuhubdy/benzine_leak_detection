# Apply SPOD filtering algorithm to sensor data 
# Create wind roses
# 12/08/15
# Halley Brantley and Dendi Suhubdy

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
import os,sys

os.chdir(os.path.dirname(sys.argv[0]))
datafolder = "./data/"
figfolder = "./report/"

minTime = pd.Timestamp(dt.datetime(2014, 11, 14, 03, 00))
maxTime = pd.Timestamp(dt.datetime(2014, 11, 14, 09, 00))

freq = '3H'
avgTime = '5min'

thresh1 = 0.08
thresh2 = 0.20
smoothingWindow = 40
FiltAvg = piecewiseImportSpod(minTime, maxTime, freq, avgTime, thresh1, thresh2, smoothingWindow, datafolder)

meanCountsPer5min = 0.017
splineBase, splineRemote, splineNone = splitBySignalButter(FiltAvg, meanCountsPer5min)
butterBase, butterRemote, butterNone = splitBySignalSpline(FiltAvg, meanCountsPer5min)

# Base Signal WindRose
if len(butterBase) > 2:
    myWindRose(butterBase['WD'], butterBase['WS'], figfolder+'windrose_Base.png')
# Remote Signal WindRose
if len(butterRemote) > 2:
    myWindRose(butterRemote['WD'], butterRemote['WS'], figfolder+'windrose_Remote.png')
# No Signal WindRose
if len(butterNone) > 2:
    myWindRose(butterNone['WD'], butterNone['WS'], figfolder+'windrose_None.png')
