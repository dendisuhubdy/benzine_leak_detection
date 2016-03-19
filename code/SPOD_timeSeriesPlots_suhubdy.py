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
plt.close('all')
# Time Series Plots
#datafolder = "/Users/halleybrantley/Desktop/computing/final/Data"
#figfolder = "/Users/halleybrantley/Desktop/Dropbox/ST758_final/figures/"
datafolder = "/Users/dendisuhubdy/Google Drive/NCSU/fall2015/st758/finalproject/datanovember/"
figfolder = "/Users/dendisuhubdy/Dropbox/ST758_final/figures/"
startTime = dt.datetime(2014, 11, 15, 00, 00)
endTime = dt.datetime(2014, 11, 17, 00, 00)
minTime = pd.Timestamp(startTime)
maxTime = pd.Timestamp(endTime)


rawdat = importSPOD(datafolder, 1, minTime, maxTime)
rawdat['timeStamp'] = pd.Series(pd.date_range(minTime, maxTime, freq='10s'), index=pd.date_range(minTime, maxTime, freq='10s')).resample('1s', fill_method = 'pad')

font = {'weight' : 'bold',
        'size'   : 6}
mpl.rcParams['axes.xmargin'] = .25
mpl.rc('font', **font)

base = ggplot(aes(x='timeStamp', y='Base'), data=rawdat) +\
   geom_line(color='blue') +\
   ylab('Base Sensor (V)') +\
   xlab('') + ylim(0,5.1) +\
   scale_x_date(labels='%m/%d %H:00', breaks=date_breaks('6 hours'))
   #   theme_matplotlib(mpl.rc('font', **font), matplotlib_defaults=False)
ggsave(plot = base, filename = figfolder+'Base.png', width = 8, height = 3)

remote = ggplot(aes(x='timeStamp', y='Remote'), data=rawdat) +\
    geom_line(color='blue') +\
    ylab('Remote Sensor (V)') +\
    xlab('') + ylim(0,5.1) +\
    scale_x_date(labels='%m/%d %H:00', breaks=date_breaks('6 hours'))
#   theme_matplotlib(mpl.rc('font', **font), matplotlib_defaults=False)

ggsave(plot = remote, filename = figfolder+'Remote.png', width = 8, height = 3)
