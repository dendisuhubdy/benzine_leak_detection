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

datafolder = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/data"
figfolder = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/"
#datafolder = "/Users/dendisuhubdy/Google Drive/NCSU/fall2015/st758/finalproject/datanovember/"
#figfolder = "/Users/dendisuhubdy/Dropbox/ST758_final/figures/"

minTime = pd.Timestamp(dt.datetime(2014, 11, 01, 01, 01))
maxTime = pd.Timestamp(dt.datetime(2014, 11, 1, 12, 01))

freq = '4H'
avgTime = '5min'
thresh1 = 0.08
thresh2 = 0.20
smoothingWindow = 40
FiltAvg = piecewiseImportSpod(minTime, maxTime, freq, avgTime, thresh1, thresh2, smoothingWindow, datafolder)

meanCountsPer5min = 0.017
splineBase, splineRemote, splineNone = splitBySignalButter(FiltAvg, meanCountsPer5min)
butterBase, butterRemote, butterNone = splitBySignalSpline(FiltAvg, meanCountsPer5min)
baseName = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/figures/"+"windrose_Base.png"
remoteName = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/figures/"+"windrose_Remote.png"
noneName = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/figures/"+"windrose_None.png"
# Base Signal WindRose
if len(butterBase) > 2:
    myWindRose(butterBase['WD'], butterBase['WS'], baseName)
# Remote Signal WindRose
if len(butterRemote) > 2:
    myWindRose(butterRemote['WD'], butterRemote['WS'], remoteName)
# No Signal WindRose
if len(butterNone) > 2:
    myWindRose(butterNone['WD'], butterNone['WS'], noneName)
