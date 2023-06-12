#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tarfile
import urllib
import seaborn as sns
import os 
import csv
from sklearn.datasets import make_blobs
import librosa 
from scipy import signal
from scipy.signal import find_peaks
from scipy import stats
import time 
from itertools import chain 
import matplotlib
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from astropy.stats import rayleightest
from scipy import stats


#%% some helper functions
def t2s(a):
    return librosa.time_to_samples(a)
def s2t(a):
    return librosa.samples_to_time(a)
def per2bpm(per):
    return np.round(60./(per),1)
def Hz2bpm(hz):
    return np.round(60.*hz, 2)

def mapZCs2Circle(zcs, bwindows):
    pooledzcs = []
    for osc in zcs:
        binnedzcs = binBeats(osc, bwindows)
        cmapzcs = []
        for i in range(1, len(binnedzcs[:19])):        
            zctobin = binnedzcs[i-1]
            binmin = bwindows[i-1]
            binmax = bwindows[i]
            bininterp = interp1d([binmin, binmax], [-np.pi, np.pi]) #map tap values within window from 0-2pi
            cmapzcs.append(list(bininterp(zctobin)))
        flatzcs = list(chain.from_iterable(cmapzcs)) 
        pooledzcs.extend(flatzcs)
    return np.array(pooledzcs)

# get the trigger/zcs from stimulus file
def getTrigs(trigdir):
    zcs = np.load(trigdir, allow_pickle=True)
    zcs = [s2t(elem)-10 for elem in zcs]
    zcs = [elem[elem>0] for elem in zcs]
    return zcs

# bin beats
def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins) +1)]
    return bins
# bin beats from the beat window 
def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)

            if num_taps_in_bin > 1:   
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window if multiple in one window
            if num_taps_in_bin == 0:
                binnedtaps.append(np.nan)
            if num_taps_in_bin == 1:
                binnedtaps.append(taps[i][0])
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim

def returnBB(thestim):
        
    stimsplit = thestim.split('_')
    bname = '_'.join([stimsplit[0], stimsplit[2], stimsplit[3] + '.txt'])
    bwindir = os.path.join(stimtimbredir, 'stimuli_' + stimsplit[-1], 'phases', 'beat-windows', bname)
    bwindows = np.loadtxt(bwindir)
    bwindows = s2t(bwindows)
    return bwindows

def mapTaps2Circle(bb, bwindows):
    cmaptaps = []
    for i in range(1, len(bwindows)):
        taptobin = bb[i-1]
        binmin = bwindows[i-1]
        binmax = bwindows[i]
        bininterp = interp1d([binmin, binmax], [-np.pi, np.pi]) #map tap values within window from 0-2pi
        cmaptaps.append(float(bininterp(taptobin))) 
    return cmaptaps

def getCircMeanVector(mappedtaps):
    mappedtaps = np.array(mappedtaps)
    R = np.abs(np.nanmean(np.exp(1j*mappedtaps)))
    psi = np.angle(np.nanmean(np.exp(1j*mappedtaps)))
    return R, psi

# calculate the frequency normalized root-mean squared deviation (frmsd) 
# with isochronous freq grids 
# freq_res = how many steps between each Hz
def frmsd_iso(taps, shift_res=60, freq_res=30, freqlow=0.5, freqhigh=4):  
    taps = np.array(taps)
    taps = taps[taps < 19] # only take taps responses within 19 second window  
    shifts = np.linspace(0, 1, shift_res)
    freq_res = freq_res*(freqhigh-freqlow) + 1 # freq resolution between Hz 
    freqshifts = np.linspace(freqlow, freqhigh, int(freq_res))    
    frmsd_hz = []    
    for i,freq in enumerate(freqshifts):
        #print(f'working on freq {freq} %r/%r'%(i, len(freqshifts)))
        bbins = np.arange(0, 19, 1/freq)       
        frmsd = []
        for j, shiftamt in enumerate(shifts):
            #print(f'\t working on shift: {shiftamt} %r/%r' %(j, len(shifts)))
            bbinshift = bbins + shiftamt
            offset = []
            for tap in taps:
                rse = find_nearest_diff(bbinshift, tap)
                offset.append(rse)
            # get root mean squared deviation 
            offsetmx = np.nanmean(np.array(offset))
            offsetmx = np.sqrt(offsetmx)*freq
            #print(f'mx offset: {offsetmx}')
            frmsd.append(offsetmx)
        min_frmsd = min(frmsd)
        #print(f'minimum frmsd is: {min_frmsd} at freq {freq}')
        frmsd_hz.append(min_frmsd)
    
    return frmsd_hz, freqshifts

# calculate normalized Pairwise-Variability Index 
def getnPVI(itis):
    nPVI = []
    for n in range(1, len(itis)):
        iti_diff = itis[n] - itis[n-1]
        iti_diff = np.abs(iti_diff/(itis[n] + itis[n-1])/2)
        nPVI.append(iti_diff)
    nPVI = np.array(nPVI)/(len(itis)-1)
    nPVI = 100*np.nansum(nPVI)
    return nPVI
    
def reformatStim(thestim):
    thestim = thestim.split('_')
    newstimname = thestim[0] + '_' + thestim[-2] + '_' + thestim[-1] + '.txt'
    return newstimname


