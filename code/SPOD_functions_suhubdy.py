import numpy as np
import numpy.random as nr
import datetime as dt
import os
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

datafolder = "/Users/dendisuhubdy/Google Drive/NCSU/fall2015/st758/finalproject/datanovember/"
figfolder = "/Users/dendisuhubdy/Dropbox/ST758_final/figures/"
minTime = pd.Timestamp(dt.datetime(2014, 11, 1, 01, 01))
maxTime = pd.Timestamp(dt.datetime(2014, 11, 30, 0, 0))

def importSPOD(datafolder, timeRes, minTime, maxTime):
    if minTime.hour == 0:
        # Adjust date because some files begin after midnight
        minNew = minTime-pd.Timedelta(1, unit='d')
    else:
        minNew = minTime
    dayRange = pd.date_range(minNew, maxTime, freq='D')
    for i in xrange(len(dayRange)):
        name = "SENTINEL Data_" + str(dayRange.year[i]) + '-' +\
            "{:02}".format(dayRange.month[i]) + '-' +\
            "{:02}".format(dayRange.day[i]) + '.csv'
        fname = os.path.join(datafolder, name)
        if i == 0:
            sensor = pd.read_csv(fname, parse_dates = {'dateTime': ['TimeStamp']},index_col = 'dateTime')
            i += 1
        else:
            df = pd.read_csv(fname, parse_dates = {'dateTime': ['TimeStamp']},index_col = 'dateTime')
            sensor = pd.concat([sensor, df])
            i += 1
        sensor = sensor.loc[:,['Sonic.U','Sonic. V','Sonic.W','Sonic.Temp','DAQ.PID (V)','DAQ.Humidity (V)','Remote.PID (V)','Remote.Humidity (V)']]
        avgtime = str(timeRes)+'S'
        sensor_avg = sensor.resample(avgtime, how='mean')
        sensor_avg.columns = ['U', 'V', 'W', 'Temp', 'Base','Base_Humidity', 'Remote', 'Remote_Humidity']
        sensor_avg['WD'] = np.arctan2(-sensor_avg['U'], -sensor_avg['V'])*180/math.pi + 180
        sensor_avg['WS'] = (sensor_avg['U']**2 + sensor_avg['V']**2)**0.5
        sensor_avg['Time'] =  pd.to_datetime(sensor_avg.index.copy()).astype('int').astype(float)/(10**18)
        sensor_out = sensor_avg[minTime:maxTime].copy()
    return(sensor_out)

def fitMinSpline(Yvar, Xvar, smoothingWindow, plot=False, plotVar = None):
    X = np.asarray(patsy.dmatrix("cr(x, df=7)-1", {"x": Xvar}))
    modDat = pd.DataFrame(X, index=Yvar.index)
    modDat.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
    modDatTrunc = modDat.iloc[smoothingWindow/2:-smoothingWindow/2].copy()
    window = np.ones(smoothingWindow)/float(smoothingWindow)
    modDatTrunc['Y'] = np.convolve(Yvar, window, 'same')[smoothingWindow/2:-smoothingWindow/2]
    mod = smf.quantreg('Y~X1+X2+X3+X4+X5', modDatTrunc)
    res = mod.fit(q=0.01)
    preds = pd.Series(res.predict(modDat), index = Xvar.index)
    if plot:
        plotDF = pd.concat([plotVar, Yvar, preds],1)
        print(plotDF.columns)
        plotDF.columns = [plotVar.name, Yvar.name, 'fitted']
        p = ggplot(aes(x=plotVar.name, y=Yvar.name), data=plotDF) + geom_line() +\
            geom_line(aes(y='fitted'), color='red')+\
            ylim(0,5) +\
            xlab('') + ylab('Sensor (V)')
        print(p)
    return(preds)

def isSignal(x, thresh):
        if x > thresh:
            return(1)
        else:
            return(0)

def applyFilters(dat, thresh1, thresh2, smoothingWindow):
    # Butterworth bandpass filter
    N  = 2   # Filter order
    Wn = [0.01, 0.1] # Cutoff frequency
    B, A = signal.butter(N, Wn, btype='bandpass', output='ba')
    RemoteStd = dat.Remote - dat.Remote.mean()
    BaseStd = dat.Base - dat.Base.mean()
    dat['butterRemote'] = pd.Series(signal.filtfilt(B,A, RemoteStd), index=dat.index)
    dat['butterBase'] = pd.Series(signal.filtfilt(B,A, BaseStd), index=dat.index)
    dat['butterRemoteSignal'] = dat.butterRemote.apply(isSignal, args = (thresh1,))
    dat['butterBaseSignal'] = dat.butterBase.apply(isSignal, args = (thresh1,))
    # Spline Quantile Regression Filter
    print(len(dat['Remote']))
    print(len(dat['Time']))
    dat['splineRemote'] = dat['Remote'] - fitMinSpline(dat['Remote'], dat['Time'], smoothingWindow)
    dat['splineBase'] = dat['Base'] - fitMinSpline(dat['Base'], dat['Time'], smoothingWindow)
    dat['splineRemoteSignal'] = dat.splineRemote.apply(isSignal, args = (thresh2,))
    dat['splineBaseSignal'] = dat.splineBase.apply(isSignal, args = (thresh2,))
    return(dat)

def getWindAvg(FiltDat, avgTime):
    FiltAvg = FiltDat.resample(avgTime, how ='mean', label='right')
    FiltAvg['WD'] = np.arctan2(-FiltAvg.U, -FiltAvg.V)*180/math.pi +180
    return(FiltAvg)

# WindRose
def myWindRose(wd, ws, fname):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = wr.WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    ax.bar(wd, ws, opening=0.8,
        bins=np.arange(0.01,6,1), normed=True, edgecolor='black')
    l = ax.legend(borderaxespad=-0.1, title="m/s", loc=0)
    plt.setp(l.get_texts(), fontsize=12)
    plt.savefig(fname)


def piecewiseImportSpod(minTime, maxTime, freq, avgTime, thresh1, thresh2, smoothingWindow, datafolder, sim=False):
    timeRange = pd.date_range(minTime, maxTime, freq=freq)
    for i in xrange(len(timeRange)-1):
        print timeRange[i]
        if sim:
            rawdat = datafolder[timeRange[i]:timeRange[i+1]].copy()
        else:
            rawdat = importSPOD(datafolder, 1, timeRange[i], timeRange[i+1])
        FiltDat = applyFilters(rawdat, thresh1, thresh2, smoothingWindow)
        FiltAvgNew = getWindAvg(FiltDat, avgTime)
        if i == 0:
            FiltAvg = FiltAvgNew
        else:
            FiltAvg = pd.concat([FiltAvg, FiltAvgNew])
    return(FiltAvg)

def splitBySignalButter(FiltAvg, thresh):
    signalBase = FiltAvg[FiltAvg.butterBaseSignal > thresh].copy()
    signalRemote = FiltAvg[FiltAvg.butterRemoteSignal > thresh].copy()
    noSignal = FiltAvg[(FiltAvg.butterBaseSignal <= thresh)&(
        FiltAvg.butterRemoteSignal <= thresh)].copy()
    return signalBase, signalRemote, noSignal

def splitBySignalSpline(FiltAvg, thresh):
    signalBase = FiltAvg[FiltAvg.splineBaseSignal > thresh].copy()
    signalRemote = FiltAvg[FiltAvg.splineRemoteSignal > thresh].copy()
    noSignal = FiltAvg[(FiltAvg.splineBaseSignal <= thresh)&(
        FiltAvg.splineRemoteSignal <= thresh)].copy()
    return signalBase, signalRemote, noSignal

def genBrownianMotion (n, tMax=10.0):
    tSeq = np.arange (tMax/float(500),
                    tMax*(1+1/float(500)), tMax/float(500));
    sig = np.zeros ((500,500), dtype='float64');
    for i in range (500):
        sig[i,0:i] = tSeq[0:i];
        sig[i,i:] = tSeq[i];
    sigSqrt = LA.cholesky (sig, lower=True);
    for j in xrange(n/500):
        z = np.dot (sigSqrt, nr.randn (500));
        if j == 0:
            zTot = np.insert (z, 0, 0);
        else:
            z = z + zTot[-1]
            zTot = np.append(zTot, z)
    m = n % 500  - 1
    tSeq = np.arange (tMax/float(m),
                    tMax*(1+1/float(m)), tMax/float(m));
    sig = np.zeros ((m,m), dtype='float64');
    for i in range (m):
        sig[i,0:i] = tSeq[0:i];
        sig[i,i:] = tSeq[i];
    print(sig)
    sigSqrt = LA.cholesky (sig, lower=True);
    z = np.dot (sigSqrt, nr.randn (m));
    z = z + zTot[-1]
    zTot = np.append(zTot, z)
    return zTot

def genBrownianBridge (n):
    tot = n
    zTot = []
    m = 3600
    tSeq = np.arange (1/float(m), 1, 1/float(m));
    m = len (tSeq);
    sig = np.zeros ((m,m), dtype='float64');
    for i in range (m):
        sig[i,0:i] = tSeq[0:i];
        sig[i,i:] = tSeq[i];
    sig11 = sig;
    sig21 = tSeq;
    sig12 = np.transpose (sig21);
    muCond = np.zeros (m);
    sigCond = sig11 - np.outer(sig12, sig21);
    sigCondSqrt = LA.cholesky (sigCond, lower=True);
    for j in xrange(tot/3600):
        z = muCond + np.dot (sigCondSqrt, nr.randn (m));
        z = np.insert (z, 0, 0);
        if j == 0:
            zTot = z;
        else:
            zTot = np.append(zTot, z)
    n = tot % 3600 - 1
    if n > 1:
        tSeq = np.arange (1/float(n), 1, 1/float(n));
        n = len (tSeq);
        sig = np.zeros ((n,n), dtype='float64');
        for i in range (n):
            sig[i,0:i] = tSeq[0:i];
            sig[i,i:] = tSeq[i];
        sig11 = sig;
        sig21 = tSeq;
        sig12 = np.transpose (sig21);
        muCond = np.zeros (n);
        sigCond = sig11 - np.outer(sig12, sig21);
        sigCondSqrt = LA.cholesky (sigCond, lower=True);
        z = muCond + np.dot (sigCondSqrt, nr.randn (n));
        z = np.insert (z, 0, 0);
        zTot = np.append(zTot, z)
    return zTot

#myBridge = genBrownianBridge (21582)
#print(len(myBridge))
