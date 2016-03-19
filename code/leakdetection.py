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
import os
import statsmodels.formula.api as smf
import scipy.linalg as LA
import matplotlib as mpl

class leakdetection():
	def __init__(self, startdate, enddate, starthour, endhour, frequency, averagetime, threshold1, threshold2, smoothing):
            '''
            The initialization of the class
            Here we initialize with the variables for our model
            Inputs:
            startdate   = define the startdate of the month for analysis
            enddate     = define the enddate of the month for analysis
            frequency   = define the frequency hour of the analysis (1H, 2H, etc)
            averagetime = define the averaging window of our analysis
            threshold 1 = define the first threshold of cutoff
            threshold 2 = define the second threshold of cutoff
            smoothing   = define the smoothing window number
            '''
            # this is the folder where we save our date, do not change
            self._datafolder = "../data/"
            # the figures folder where it will be saved
	    self._figfolder = "../figures/"
            # define the filename for simulation results
	    self._fname = "../simulated/Simulated_07052014_5min.csv"
	    self._startdate = startdate
	    self._enddate = enddate
            self._starthour = starthour
            self._endhour = endhour
            self._minTime = pd.Timestamp(dt.datetime(2014, 11, self._startdate, self._starthour, 01))
	    self._maxTime = pd.Timestamp(dt.datetime(2014, 11, self._enddate, self._endhour, 01))
	    self._freq = frequency #'4H'
	    self._avgTime = averagetime #'5min'
	    self._thresh1 = threshold1 #0.08
	    self._thresh2 = threshold2 #0.20
	    self._smoothingWindow = smoothing #40

	def SPOD_algorithms(self, meanCountsPer5min = 0.017):
	    '''
            This is the main algorithm for analysis
            '''
            # define the filtered average using
            # piecewise import SPOD sensor
            FiltAvg = self.piecewiseImportSpod(self._minTime, self._maxTime, self._freq, self._avgTime, self._thresh1, self._thresh2, self._smoothingWindow, self._datafolder)

	    #meanCountsPer5min = 0.017
	    splineBase, splineRemote, splineNone = self.splitBySignalButter(FiltAvg, meanCountsPer5min)
	    butterBase, butterRemote, butterNone = self.splitBySignalSpline(FiltAvg, meanCountsPer5min)

	    # Base Signal WindRose
	    if len(butterBase) > 2:
		self.myWindRose(butterBase['WD'], butterBase['WS'], self._self._figfolder+'windrose_Base.png')
	    # Remote Signal WindRose
	    if len(butterRemote) > 2:
		self.myWindRose(butterRemote['WD'], butterRemote['WS'], self._figfolder+'windrose_Remote.png')
	    # No Signal WindRose
	    if len(butterNone) > 2:
		self.myWindRose(butterNone['WD'], butterNone['WS'], self._figfolder+'windrose_None.png')

	def importSPOD(self, datafolder, timeRes, minTime, maxTime):
	    if self._minTime.hour == 0:
	        # Adjust date because some files begin after midnight
	        minNew = self._minTime-pd.Timedelta(1, unit='d')
	    else:
	        minNew = self._minTime
	    dayRange = pd.date_range(minNew, self._maxTime, freq='D')
	    for i in xrange(len(dayRange)):
                # loop through the dayrange for string filename parsing
	        name = "SENTINEL Data_" + str(dayRange.year[i]) + '-' +"{:02}".format(dayRange.month[i]) + '-' +"{:02}".format(dayRange.day[i]) + '.csv'
	        # the defined name is in the format SENTINEL Data_year-month-day.csv
                self._fname = os.path.join(self._datafolder, name)
	        if i == 0:
	            # if the it is the first row
                    # then read the CSV
                    sensor = pd.read_csv(self._fname, parse_dates = {'dateTime': ['TimeStamp']},index_col = 'dateTime')
	            # go to the next row
                    i += 1
	        else:
                    # define the dataframe for analysis
                    # define parse dates
                    df = pd.read_csv(self._fname, parse_dates = {'dateTime': ['TimeStamp']},index_col = 'dateTime')
	            # concatenate sensor dataframe
                    sensor = pd.concat([sensor, df])
	            # go to the new row of data
                    i += 1
	        sensor = sensor.loc[:,['Sonic.U','Sonic. V','Sonic.W','Sonic.Temp','DAQ.PID (V)','DAQ.Humidity (V)','Remote.PID (V)','Remote.Humidity (V)']]
	        # define the sensor data that is needed for analysis
                # those are Sonic.U, Sonic.V, Sonic.W, Sonic.Temp, PID, Humity on both base and remote sensors
                self._avgTime = str(timeRes)+'S'
	        sensor_avg = sensor.resample(self._avgTime, how='mean')
                # resampling using given statistical analysis
                # we choose the mean
	        sensor_avg.columns = ['U', 'V', 'W', 'Temp', 'Base','Base_Humidity', 'Remote', 'Remote_Humidity']
	        # average WD
                sensor_avg['WD'] = np.arctan2(-sensor_avg['U'], -sensor_avg['V'])*180/math.pi + 180
	        # average WS
                sensor_avg['WS'] = (sensor_avg['U']**2 + sensor_avg['V']**2)**0.5
	        # average Time
                sensor_avg['Time'] =  pd.to_datetime(sensor_avg.index.copy()).astype('int').astype(float)/(10**18)
	        # average out sensors
                sensor_out = sensor_avg[self._minTime:self._maxTime].copy()
	    return(sensor_out)

	def fitMinSpline(self, Yvar, Xvar, smoothingWindow, plot=False, plotVar = None):
	    '''
            This function is to fit/interpolate a spline in the data
            '''
            # use patsy class to define a matrix
            X = np.asarray(patsy.dmatrix("cr(x, df=7)-1", {"x": Xvar}))
	    # redefine dataframe
            modDat = pd.DataFrame(X, index=Yvar.index)
	    # redefine our data into X1-X7
            modDat.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
	    modDatTrunc = modDat.iloc[self._smoothingWindow/2:-self._smoothingWindow/2].copy()
	    window = np.ones(self._smoothingWindow)/float(self._smoothingWindow)
	    modDatTrunc['Y'] = np.convolve(Yvar, window, 'same')[self._smoothingWindow/2:-self._smoothingWindow/2]
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
                #return regression predictors
	    return(preds)

	def isSignal(self,x, thresh):
            '''
            This is the function for cutting of the threshold
            '''
	    if x > thresh:
	        return(1)
	    else:
	        return(0)

	def applyFilters(self,dat, thresh1, thresh2, smoothingWindow):
	    '''
            Butterworth bandpass filter analysis
            '''
	    N  = 2
		# Filter order
	    Wn = [0.01, 0.1]
		# Cutoff self._frequency
	    B, A = signal.butter(N, Wn, btype='bandpass', output='ba')
	    RemoteStd = dat.Remote - dat.Remote.mean()
	    BaseStd = dat.Base - dat.Base.mean()
	    dat['butterRemote'] = pd.Series(signal.filtfilt(B,A, RemoteStd), index=dat.index)
	    dat['butterBase'] = pd.Series(signal.filtfilt(B,A, BaseStd), index=dat.index)
	    dat['butterRemoteSignal'] = dat.butterRemote.apply(self.isSignal, args = (self._thresh1,))
	    dat['butterBaseSignal'] = dat.butterBase.apply(self.isSignal, args = (self._thresh1,))
	    # Spline Quantile Regression Filter
	    print(len(dat['Remote']))
	    print(len(dat['Time']))
	    dat['splineRemote'] = dat['Remote'] - self.fitMinSpline(dat['Remote'], dat['Time'], self._smoothingWindow)
	    dat['splineBase'] = dat['Base'] - self.fitMinSpline(dat['Base'], dat['Time'], self._smoothingWindow)
	    dat['splineRemoteSignal'] = dat.splineRemote.apply(self.isSignal, args = (self._thresh2,))
	    dat['splineBaseSignal'] = dat.splineBase.apply(self.isSignal, args = (self._thresh2,))
	    return(dat)

	def getWindAvg(self, FiltDat, avgTime):
            '''
            Obtain window averaging with the resampling data
            '''
	    FiltAvg = FiltDat.resample(self._avgTime, how ='mean', label='right')
	    FiltAvg['WD'] = np.arctan2(-FiltAvg.U, -FiltAvg.V)*180/math.pi +180
	    return(FiltAvg)

	def myWindRose(self, wd, ws, fname):
            '''
            Define Windrose analysis window
            '''
	    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
	    rect = [0.1, 0.1, 0.8, 0.8]
	    ax = wr.WindroseAxes(fig, rect, axisbg='w')
	    fig.add_axes(ax)
	    ax.bar(wd, ws, opening=0.8,
	        bins=np.arange(0.01,6,1), normed=True, edgecolor='black')
	    l = ax.legend(borderaxespad=-0.1, title="m/s", loc=0)
	    plt.setp(l.get_texts(), fontsize=12)
	    plt.savefig(self._fname)


	def piecewiseImportSpod(self, minTime, maxTime, freq, avgTime, thresh1, thresh2, smoothingWindow, datafolder, sim=False):
	    '''
            This function will import the data with filters
            '''
            timeRange = pd.date_range(minTime, maxTime, None,freq)
	    for i in xrange(len(timeRange)-1):
	        print timeRange[i]
	        if sim:
	            rawdat = self._datafolder[timeRange[i]:timeRange[i+1]].copy()
	        else:
	            rawdat = self.importSPOD(datafolder, 1, timeRange[i], timeRange[i+1])
	        FiltDat = self.applyFilters(rawdat, thresh1, thresh2, smoothingWindow)
	        FiltAvgNew = self.getWindAvg(FiltDat, avgTime)
	        if i == 0:
	            FiltAvg = FiltAvgNew
	        else:
	            FiltAvg = pd.concat([FiltAvg, FiltAvgNew])
	    return(FiltAvg)

	def splitBySignalButter(self, FiltAvg, thresh):
            '''
            Here we split the signal butter with the filered average and threshold
            The signal above the threshold will be cutoff
            '''
	    signalBase = FiltAvg[FiltAvg.butterBaseSignal > thresh].copy()
	    signalRemote = FiltAvg[FiltAvg.butterRemoteSignal > thresh].copy()
	    noSignal = FiltAvg[(FiltAvg.butterBaseSignal <= thresh)&(
	        FiltAvg.butterRemoteSignal <= thresh)].copy()
	    return signalBase, signalRemote, noSignal

	def splitBySignalSpline(self, FiltAvg, thresh):
            '''
            Here we split the signal with the spline data
            If it is above the threshold then we cut itoff
            '''
	    signalBase = FiltAvg[FiltAvg.splineBaseSignal > thresh].copy()
	    signalRemote = FiltAvg[FiltAvg.splineRemoteSignal > thresh].copy()
	    noSignal = FiltAvg[(FiltAvg.splineBaseSignal <= thresh)&(
	        FiltAvg.splineRemoteSignal <= thresh)].copy()
	    return signalBase, signalRemote, noSignal

	def genBrownianMotion(self, n, tMax=10.0):
            '''
            Here we simulate a brownian motion for our bandpassfilter algorithm
            '''
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

	def genBrownianBridge(self, n):
            '''
            Defining a brownian bridge for our bandpass filter
            '''
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

	def simulatedata(self):
            '''
            simulate the data and plotting
            '''
	    nr.seed(seed=79819)
	    plt.close('all')

	    QUIC20 = pd.read_csv(self._fname)
	    #startTime = pd.Timestamp(dt.datetime(2014, 07, 05, 12, 00, 20))
	    #endTime = pd.Timestamp(dt.datetime(2014, 07, 05, 18,  00, 00))
	    #startTime = pd.Timestamp(dt.datetime(2014, 11, 01, 00, 00, 00))
	    #endTime = pd.Timestamp(dt.datetime(2014, 11, 31, 00,  00, 00))
	    TimeStamp = pd.date_range(self._minTime, self._endTime, freq='20s')
	    QUIC20['dateTime'] = pd.date_range(self._minTime, self._endTime, freq='20s')
	    QUIC20.BaseSim = QUIC20.BaseSim*10**7
	    QUIC20.RemoteSim = QUIC20.RemoteSim*10**7
	    QUIC20.index = QUIC20.dateTime
	    QUIC = QUIC20.resample('1s', fill_method = 'pad')
	    QUIC['dateTime'] = QUIC.index.copy()
	    rawdat = self.importSPOD(self._datafolder, 1, self._minTime, self._maxTime)
	    QUIC['U'] = rawdat['U']
	    QUIC['V'] = rawdat['V']
	    QUIC['WS'] = rawdat['WS']
	    QUIC['Time'] =  pd.to_datetime(QUIC.index.copy()).astype('int').astype(float)/(10**18)

	    # Simulate Data
	    Num = len(QUIC)
	    QUIC['BaseBM'] = self.genBrownianBridge (Num) + 1.5

	    NewBase  = QUIC.BaseSim + QUIC.BaseBM
	    NewBase[NewBase > 5] = 5
	    NewBase[NewBase < 0.25] = nr.randn(len(NewBase[NewBase < 0.25]))*.05 + 0.25
	    QUIC['Base'] = NewBase

	    QUIC['RemoteBM'] = self.genBrownianBridge (Num)
	    NewRemote  = QUIC.RemoteSim + QUIC.RemoteBM
	    NewRemote[NewRemote > 5] = 5
	    NewRemote[NewRemote < 0.25] = nr.randn(len(NewRemote[NewRemote < 0.25]))*.05 + 0.25
	    QUIC['Remote'] = NewRemote

	    # Plot Simulated Data
	    font = {'weight' : 'bold', 'size'   : 8}
	    mpl.rc('font', **font)

	    baseTotal = ggplot(aes(x='dateTime', y='Base'), data=QUIC) +geom_line() +ylim(0,5) +geom_line() + xlab("") + ylab("Simulated Signal (V)")

	    baseRand = ggplot(aes(x='dateTime', y='BaseBM'), data=QUIC) +\
		geom_line() + xlab("") + ylab("Stochastic Baseline")

	    baseSim = ggplot(aes(x='dateTime', y='BaseSim'), data=QUIC) +\
		geom_line()+\
		ylim(0,5) + xlab("") + ylab("Simulated Signal (V)")

	    remoteTotal = ggplot(aes(x='dateTime', y='Remote'), data=QUIC) +\
		geom_line() + xlab("") + ylab("Simulated Signal (V)")
	     #   theme_matplotlib(mpl.rc('font', **font), matplotlib_defaults=False)

	    ggsave(plot = baseTotal, filename = self._figfolder+'BaseTotal.png', width = 8, height = 2)
	    ggsave(plot = baseRand, filename = self._figfolder+'BaseRand.png', width = 8, height = 2)
	    ggsave(plot = baseSim, filename = self._figfolder+'BaseSim.png', width = 8, height = 2)
	    ggsave(plot = remoteTotal, filename = self._figfolder+'remoteTotal.png', width = 8, height = 2)

	    # Illustrate Method
	    self.fitMinSpline(QUIC['Base'][QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')],
	    QUIC['Time'][QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')], self._smoothingWindow, plot=True, plotVar = QUIC.dateTime)
	    ggsave(filename = self._figfolder+'Spline_fit.png', width = 8, height = 2)

	    QUICFilt = self.applyFilters(QUIC[QUIC.index.min():QUIC.index.min()+pd.Timedelta(freqT, 'h')], self._thresh1, self._thresh2, self._smoothingWindow)
	    butterplot = ggplot(aes(x='dateTime', y='butterBase'), data=QUICFilt) + geom_line() + ylim(0,5) + xlab('') + ylab('Sensor after Butterworth')
	    ggsave(plot=butterplot, filename = self._figfolder+'Butterworth_filt.png', width = 8, height = 2)

	    # Apply algorithm
	    QUIC['TrueBase'] = QUIC.BaseSim.apply(self.isSignal, args = (0.01,))
	    QUIC['TrueRemote'] = QUIC.RemoteSim.apply(self.isSignal, args = (0.01,))

	    FiltAvg = self.piecewiseImportSpod(self._minTime, self._maxTime, self._freq, self._avgTime, self._thresh1, self._thresh2, self._smoothingWindow, QUIC, True)

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

	    ggsave(plot = TrueVDetect, filename = self._figfolder+'TrueVDetect.png', width = 4.5, height = 4)

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

	def timeseriesplots(self):
	    rawdat = self.importSPOD(self._datafolder, 1, self._minTime, self._maxTime)
	    rawdat['timeStamp'] = pd.Series(pd.date_range(self._minTime, self._maxTime, freq='10s'), index=pd.date_range(self._minTime, self._maxTime, freq='10s')).resample('1s', fill_method = 'pad')

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
	    ggsave(plot = base, filename = self._figfolder+'Base.png', width = 8, height = 3)

	    remote = ggplot(aes(x='timeStamp', y='Remote'), data=rawdat) +\
		    geom_line(color='blue') +\
		    ylab('Remote Sensor (V)') +\
		    xlab('') + ylim(0,5.1) +\
		    scale_x_date(labels='%m/%d %H:00', breaks=date_breaks('6 hours'))
	    #   theme_matplotlib(mpl.rc('font', **font), matplotlib_defaults=False)

	    ggsave(plot = remote, filename = self._figfolder+'Remote.png', width = 8, height = 3)

def main():
    # here we define the class in the variable myleak
    # define the startdate = 2 Nov, enddate =11 Nov, averaging window= 4 hours, threshold1 = 0.08, threshold2 = 0.2, and smoothing window of 40
    myleak = leakdetection(5, 7, 12,18,'4H', '5min', 0.08, 0.20, 40)
    # run the main algorithm
    myleak.SPOD_algorithms()
    # simulate the data
    myleak.Simulate()
    # plot the timeseries plot
    myleak.timeseriesplots()
    return

if __name__ == "__main__":
    main()
