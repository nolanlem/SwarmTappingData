#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:14:56 2022

@author: nolanlem
"""

import numpy as np 
import pandas as pd
import glob 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker
import sys
import os
from ast import literal_eval
from io import StringIO
import itertools
from scipy.signal import find_peaks
from collections import defaultdict
import librosa
from scipy.stats import sem
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.cm as cm
import datetime 
from collections import defaultdict
import scipy.stats
import csv


import seaborn as sns
sns.set()
from fun.functions import *


sr=22050


def removeStrFormatting(str_array):
    for str_arr in str_array:
        str_arr = str_arr[1:-1] # remove "'[" and "]'"
        str_arr = str.split(str_arr, ',') # split strings
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
        #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr


def flatten2DList(thelist):
    flatlist = list(itertools.chain(*thelist))
    return flatlist

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    for i, tap in enumerate(taps):
        try:
            binnedtaps.append(taps[i][0]) # take first tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    return binnedtaps

def removeStrFormatting(str_arr):
    str_arr = str_arr[1:-1] # remove "'[" and "]'"
    str_arr = str.split(str_arr, ',') # split strings
    try:
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
    except ValueError:
        pass
    #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr

def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)
        
def find_nearest_diff(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    diff = np.abs(array[idx] - value)
    diff_sq = diff**2
    return diff_sq
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

def per2bpm(per):
    return np.round(60./(per),1)

def Hz2bpm(hz):
    return np.round(60.*hz, 2)

def reformatStim(thestim):
    thestim = thestim.split('_')
    newstimname = thestim[0] + '_' + thestim[-2] + '_' + thestim[-1] + '.txt'
    return newstimname

########### get CENTER PERIODS and BEAT BINS from data txt files from generative model ########
##################################################################################################

def load_subjects_resps():
    # dictionaries for center periods, beat bins, etc. from generative model (already generated) 
    idealperiods, sndbeatbins, centerbpms, centerperiods = {},{},{},{}

    # load beatbins for no-timbre type
    datadirs = ['./stim-no-timbre-5', './stim-timbre-5']
    timbre_tags = ['n','t']
    stimuli_dirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']
    #beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
    #centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')

    for datadir, ttag in zip(datadirs, timbre_tags):
        for stimuli_dir in stimuli_dirs:
            beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
            for fi in glob.glob(beatbins_dir + '/*.txt'):
                fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
                f = str.split(fi_basename, '_') 
                sync_cond = "_".join([f[0], ttag, f[1], f[2]])
                sndbeatbins[sync_cond] = s2t(np.loadtxt(fi, delimiter='\n'))
                
                bwindows = s2t(sndbeatbins[sync_cond])
                biti = bwindows[:-1] + np.diff(bwindows)/2
                avgbpm = np.mean(np.diff(biti))
                #cbpm[stim] = np.loadtxt(cbpmpath)
                centerperiods[sync_cond] = avgbpm                 
                
            ### NB: the saved centerbpms are WRONG, have to recalculate based off of beatwindows above 
            # centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')    
            # for fi in glob.glob(centerbpms_dir + '/*.txt'):       
            #     fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
            #     f = str.split(fi_basename, '_') 
            #     sync_cond = "_".join([f[0], ttag, f[1], f[2]])    
            #     thecenterbpm = np.loadtxt(fi)        
            #     centerbpms[sync_cond] = float(thecenterbpm)
            #     centerperiods[sync_cond] = per2bpm(60./float(thecenterbpm))
                

    ######### parse subject taps in csv output files and format into dataframes or arrays

    # default dictionarya
    subject_resps = defaultdict(lambda: defaultdict(list))

    ordered_subjects = []

    # STRING PROMPTS in HEADER of csv files 
    block1taps = 'block1_taps.rt'
    block2taps = 'block2_taps.rt'
    csv_sndfiles = 'sndfile'
    csv_tempo = 'tempo'
    csv_coupling_cond = 'cond'
    csv_version = 'version'
    csv_participant = 'Participant Initials'
    csv_type = 'type'

    ##################################################
    ######### only take good USABLE csv files #####
    #################################################
    batch_folder = 'usable-batch-12-7'
    #batch_folder = 'usable-stanford-batch'
    #batch_folder = 'usable-mturk-batch'
    subject = []
    csvfiles = []   

    ### fill up subject with csv basename 
    for csv_ in glob.glob('./mturk-csv/' + batch_folder + '/*.csv'):
        #namestripped = os.path.basename(csv_).split('.')[0].split(' ')[0]
        namestripped = os.path.basename(csv_)
        subject.append(namestripped)
        csvfiles.append(csv_)

    ########## GET ALL STIM NAMES with full path from allstims dir --> allstims list
    allstims = []   # allstims is full file path of every stimuli 
    for fi in glob.glob('./allstims/*.wav'):
        allstims.append(fi)
    ########## READ IN THE SUBJECT TAPS #################

    for csv_file, person in zip(csvfiles, subject):
        print('SUBJECT: ', person)
        df_block = pd.read_csv(csv_file, keep_default_na=False)
        subject_resps[person] = {}  

        try:

            df_block_1 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block1taps])[4:44]
            df_block_2 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block2taps])[44:-1]
            
            df_block_1_type = df_block_1[csv_type]
            #timbre_type = df_block_1['sndfile'].values
        
            for index, row in df_block_1.iterrows():
                sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
                subject_resps[person][sync_cond_version] = []
            for index, row in df_block_2.iterrows():
                sync_cond_version  = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
                subject_resps[person][sync_cond_version] = []
        
            for index, row in df_block_1.iterrows():
                sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
                subject_resps[person][sync_cond_version] = removeStrFormatting(row[block1taps])
            for index, row in df_block_2.iterrows():
                sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
                subject_resps[person][sync_cond_version] = removeStrFormatting(row[block2taps])
        
        except TypeError:
            print('could not read %r csv file' %(person))
    return subject_resps, sndbeatbins
        
#####NB: subject_resps are now in this format 
#### subject_resps[person][type(no, timbre)][sync_tempo_version]   

