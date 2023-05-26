#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:38:35 2020

@author: nolanlem
"""


import numpy as np 
import pandas as pd
import glob 
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
import astropy
from astropy.stats import rayleightest

import seaborn as sns
sns.set()


#%%
rootdir = './'

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
        
def impute_mx(csv_file):
    rootdir = os.path.dirname(csv_file)
    df = pd.read_csv(csv_file)
    nan_mx_idx = df[df['mx'].isnull()].index
    nan_sx_idx = df[df['sx'].isnull()].index
    
    zero_mx_idx = df[df['mx'] == 0.0].index
    zero_sx_idx = df[df['sx'] == 0.0].index
    
    print('zero mx:', zero_mx_idx)
    print('zero sx:', zero_sx_idx)
    ##################
    nan_subjs = df.iloc[list(nan_mx_idx.values)]
    nan_subjects = []
    
    for subj in nan_subjs.values:
        nan_subjects.append(subj[0])
        
    num_nans_per_subject = {i:nan_subjects.count(i) for i in nan_subjects}
    print(num_nans_per_subject)
    #################
    df.insert(df.shape[1], "mx_imputed", 0.0)
    df.insert(df.shape[1], "sx_imputed", 0.0)
    df.insert(df.shape[1], "good", 1)
    ##################
    idx_nans = []
    
    for particip in nan_subjects: 
        idx_nans = df[df['subject'] == particip].index
    
    for idx in idx_nans:
        df.loc[idx, 'good'] = 0

    mx_sum = df.loc[df['mx'] > 0.0, ['mx']].sum(axis=0)
    mx_sum = mx_sum/(len(df)-len(zero_mx_idx))
    
    sx_sum = df.loc[df['sx'] != 0.0, ['sx']].sum(axis=0)
    sx_sum = sx_sum/(len(df)-len(zero_sx_idx))

    mxs, sxs = [], []
    for row in df.index:
        if df.loc[row, 'mx'] > 0.0:
            #mxs.append(df.loc[row,'mx'])
            df.loc[row, 'mx_imputed'] = df.loc[row, 'mx']
        if df.loc[row, 'mx'] ==  0.0:
            df.loc[row, 'mx_imputed'] = float(mx_sum)
        
        if df.loc[row, 'sx'] > 0.0:
            df.loc[row, 'sx_imputed'] = df.loc[row, 'sx']
        if df.loc[row, 'sx'] == 0.0:
            df.loc[row, 'sx_imputed'] = float(sx_sum)
    df.to_csv(rootdir + '/' + os.path.basename(csv_file).split('.')[0] + '-imputed.csv')
    print('creating %r' %(rootdir + '/' + os.path.basename(csv_file).split('.')[0] + '-imputed.csv'))

#%% STIM BLOCK LAYOUT 
# A1 no(1,2)        timbre(1,2)
# B1 timbre(1,2)    no(1,2)
# A2 no(3,4)        timbre(3,4)
# B2 timbre(3,4)    no(3,4)

# some help functions 
def t2s(a):
    return librosa.time_to_samples(a)
def s2t(a):
    return librosa.samples_to_time(a)
def per2bpm(per):
    return np.round(60./(per),1)
def Hz2bpm(hz):
    return np.round(60.*hz, 2)


#%%########### get CENTER PERIODS and BEAT BINS from data txt files from generative model ########
##################################################################################################

# dictionaries for center periods, beat bins, etc. from generative model (already generated) 
sr_audio = 22050.
sndbeatbins, centerbpms, centerperiods = {},{},{}

# load beatbins for no-timbre type
#beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
#centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')


for fi in glob.glob('./bb-exp-1/*.npy'):    
    sync_cond = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
    sndbeatbins[sync_cond] = np.load(fi)/sr_audio
    centerperiods[sync_cond] = np.mean(np.diff(sndbeatbins[sync_cond]))
    centerbpms[sync_cond] = 60./np.mean(np.diff(sndbeatbins[sync_cond]))
    
#%% ######### parse subject taps in csv output files and format into dataframes or arrays

# default dictionarya
subject_resps = defaultdict(lambda: defaultdict(list))

ordered_subjects = []

# STRING PROMPTS in HEADER of csv files 
block1taps = 'key_resp_9.rt'
block2taps = 'key_resp_10.rt'
csv_sndfiles = 'sndfile'
csv_tempo = 'tempo'
csv_coupling_cond = 'cond'
csv_participant = 'participant'


##################################################
######### only take good USABLE csv files #####
#################################################

#batch_folder = 'usable-stanford-batch'
#batch_folder = 'usable-mturk-batch'
subject = []
csvfiles = []  

tapdir = '../tap-data/' 

### fill up subject with csv basename 
for csv_ in glob.glob(tapdir + '*.csv'):
    #namestripped = os.path.basename(csv_).split('.')[0].split(' ')[0]
    namestripped = os.path.basename(csv_)
    subject.append(namestripped)
    csvfiles.append(namestripped)

########## GET ALL STIM NAMES with full path from allstims dir --> allstims list
allstims = []   # allstims is full file path of every stimuli 

for fi in glob.glob('../stimuli/' + '/*.mp3'):
    fi_ = os.path.basename(fi).split('.')[0]
    allstims.append(fi_)

#%%########## READ IN THE SUBJECT TAPS #################

for csv_file, person in zip(csvfiles, subject):
    print('SUBJECT: ', person)
    df_block = pd.read_csv(os.path.join(tapdir + csv_file), keep_default_na=False)
    subject_resps[person] = {}  

    try:

        df_block_1 = df_block.get([csv_participant, csv_sndfiles, csv_coupling_cond, csv_tempo, block1taps])[7:57]
        df_block_2 = df_block.get([csv_participant, csv_sndfiles, csv_coupling_cond, csv_tempo, block2taps])[57:112]
        

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
        
       
#####NB: subject_resps are now in this format 
#### subject_resps[person][type(no, timbre)][sync_tempo_version]            

#%% ### reformat trials subjects did not perform with empty list '' -> [] ###
  
subjectplotdir = os.path.join(tapdir, "subjects/")

# replace all empty trials with [] (tried with np.nan but not good for plotting... )
for person in subject:
    print(person)
    for n, sndfile in enumerate(allstims):
        sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]  
        try:
            if (subject_resps[person][sync_cond_version] == ['']):
                subject_resps[person][sync_cond_version] = []
        except KeyError:
            print('subject %r did not tap to %r' %(person, sndfile))



#%%
################################################################%%
####################################################
############ ITI ANALYSIS ##############################

#%% MUST INTIALIZE DICTIONARIES BEFORE LOOP 
### NB: MUST DO THIS EVERYTIME BEFORE RUNNING ITI ANALYSIS depending on no bb, bb, or outlier algo 

sndfile_strs = 'No Timbre Condition' # e.g. 'Timbre Condition', "No Timbre Condition"
sync_strs = ['strong','medium','weak']   # strings for coupling cond

binned_subject_taps = {}
subject_sync_cond_taps, subject_sync_cond_itis = {}, {}

for person in subject:
    subject_sync_cond_taps[person], subject_sync_cond_itis[person] = {}, {}
    for sync_str in sync_strs:
        subject_sync_cond_taps[person][sync_str], subject_sync_cond_itis[person][sync_str] = [],[]

#%% !!!!! NB:########### ONE THIS CODE BLOCK OR THE NEXT !!!!!! ##########
####### 1. BEAT BINNING
####### 2. NO BEAT BINNING 
####### 3. OUTLIERS ALGO
#######DONT FORGET
###### HAVE TO INITIALIZE DICTIONARIES IN CODE BLOCK BEFORE!!!!
#################  1. BEAT BINNING:  GET ITIs FROM SUBJECT_RESPS[] WITH BEAT BINNING 
#%%% ########### UTIL FUNCTIONS FOR BEAT BINNING ########
# def binBeats(taps, beat_bins):
#     taps = np.array(taps)
#     digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
#     bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
#     return bins

# def binTapsFromBeatWindow(taps):
#     binnedtaps = []
#     avg_taps_per_bin = []
#     for i, tap in enumerate(taps):
#         try:
#             num_taps_in_bin = len(taps[i])
#             avg_taps_per_bin.append(num_taps_in_bin)
#             if num_taps_in_bin > 0:            
#                 random_tap = np.random.randint(low=0, high=num_taps_in_bin)
#                 binnedtaps.append(taps[i][random_tap]) # take random tap in window
#         except IndexError:
#             binnedtaps.append(np.nan)
    
#     avg_taps_per_stim = np.mean(avg_taps_per_bin)
#     return binnedtaps, avg_taps_per_stim

  
#%% concatenate all the stimulus filenames
n_weak = ['./stimuli_1/tmp/vweak_110_1.wav',
 './stimuli_1/tmp/vweak_90_2.wav',
 './stimuli_1/tmp/weak_105_1.wav',
 './stimuli_1/tmp/vweak_100_1.wav',
 './stimuli_1/tmp/vweak_110_2.wav',
 './stimuli_2/tmp/medium_105_3.wav',
 './stimuli_1/tmp/medium_100_2.wav',
 './stimuli_1/tmp/weak_100_1.wav',
 './stimuli_1/tmp/medium_105_2.wav',
 './stimuli_1/tmp/medium_110_2.wav',
 './stimuli_2/tmp/vweak_100_4.wav',
 './stimuli_2/tmp/weak_100_3.wav']

n_medium = ['./stimuli_2/tmp/medium_95_4.wav',
 './stimuli_1/tmp/vweak_90_1.wav',
 './stimuli_1/tmp/vweak_95_1.wav',
 './stimuli_1/tmp/medium_95_1.wav',
 './stimuli_1/tmp/weak_95_1.wav',
 './stimuli_1/tmp/vweak_105_1.wav',
 './stimuli_1/tmp/weak_110_1.wav',
 './stimuli_1/tmp/vweak_105_2.wav',
 './stimuli_1/tmp/medium_100_1.wav',
 './stimuli_2/tmp/weak_95_4.wav',
 './stimuli_2/tmp/weak_90_4.wav',
 './stimuli_2/tmp/weak_100_4.wav']

n_strong = [
 './stimuli_2/tmp/medium_95_3.wav',
 './stimuli_1/tmp/strong_105_2.wav',
 './stimuli_2/tmp/strong_100_3.wav',
 './stimuli_2/tmp/strong_110_3.wav',
 './stimuli_1/tmp/medium_90_1.wav',
 './stimuli_2/tmp/strong_105_4.wav',
 './stimuli_1/tmp/strong_90_1.wav',
 './stimuli_1/tmp/strong_100_2.wav',
 './stimuli_1/tmp/strong_95_1.wav',
 './stimuli_2/tmp/strong_90_3.wav',
 './stimuli_1/tmp/strong_90_2.wav',
 './stimuli_2/tmp/strong_90_4.wav']

n_perfect = [
"./stimuli_1/perfect_90_0.wav",
"./stimuli_1/perfect_90_1.wav",
"./stimuli_1/perfect_95_0.wav",
"./stimuli_1/perfect_95_1.wav",
"./stimuli_1/perfect_100_0.wav",
"./stimuli_1/perfect_100_1.wav",
"./stimuli_1/perfect_105_0.wav",
"./stimuli_1/perfect_105_1.wav"
]

# reformated here bc dir change
n_weak = [os.path.basename(fi).split('.')[0] for fi in n_weak]
n_medium = [os.path.basename(fi).split('.')[0] for fi in n_medium]
n_strong = [os.path.basename(fi).split('.')[0] for fi in n_strong]
n_perfect = [os.path.basename(fi).split('.')[0] for fi in n_perfect]


allconditions = [n_strong, n_medium, n_weak]

#%% plot per subject nITI histograms 
fig, ax = plt.subplots(nrows=len(subject), ncols=1, figsize=(5,10), sharex=True, sharey=True)
bins = np.linspace(0, 1.5, 100)

itipoolsubj = {}
for i, person in enumerate(subject):
    itipoolsubj[person] = {}
    for sync_cond, sync_str in zip(allconditions, sync_strs):
        itipoolsubj[person][sync_str] = []
        for sync_cond_version in sync_cond:
            taps = subject_resps[person][sync_cond_version]
            if len(taps) > 10:
                taps = taps[:10]
            iti = np.diff(taps)/centerperiods[sync_cond_version]
            itipoolsubj[person][sync_str].extend(iti)
        ax[i].hist(itipoolsubj[person][sync_str], bins=bins, alpha=0.5, label=sync_str)
        ax[i].set_xlim([0,1.5])
        personstr = person[:2]
        ax[i].set_title(f'{personstr}', fontsize=8)

plt.tight_layout()
ax[i].legend()
fig.suptitle(f'exp 1 subject asynchronies')
ax[i].set_xlabel('normalized ITI')
# fig.savefig(os.path.join(rootdir, 'plots', 'asynchronies', f'exp 1 subject asynchronies.png'), dpi=120)
#%% separate by tap strategy group 
# group converters and loyalists (manually)
converters = [subject[1], subject[7]] # converters is 'fast' group 
loyalists = [subj for subj in subject if subj not in converters] # loyalists is 'regular' group 

#%% make aggregate nITI histogram plot
loyalist_itipool, converter_itipool, itipool = {}, {}, {}
for sync_cond, sync_str in zip(allconditions, sync_strs):
    print(f'working on {sync_str}')
    loyalist_itipool[sync_str], converter_itipool[sync_str] = [], []
    for person in loyalists:
        for sync_cond_version in sync_cond:
            taps = subject_resps[person][sync_cond_version]
            if len(taps) > 10:
                taps = taps[:10]
            iti = np.diff(taps)/centerperiods[sync_cond_version]
            loyalist_itipool[sync_str].extend(iti)
    
    for person in converters:
        for sync_cond_version in sync_cond:
            taps = subject_resps[person][sync_cond_version]
            if len(taps) > 10:
                taps = taps[:10]
            iti = np.diff(taps)/centerperiods[sync_cond_version]
            converter_itipool[sync_str].extend(iti)            
            
#%%            
##### PLOT HISTOGRAM OF AGGREGATED ASYNCHRONIES PER COUPLING AND SUBJECT CAT 
fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(10,10))

bins = np.linspace(0, 1.5, 100)


for i, sync_str in enumerate(sync_strs):
    color_ = sns.color_palette()[i]
    ax[i].hist(loyalist_itipool[sync_str], bins=bins, alpha=0.5, color=color_, label='Regular Tapping Group (N=8)') # loyalists  
    amt = min(color_)
    paler_color = (color_[0]-amt, color_[1]-amt, color_[2])
    ax[i].hist(converter_itipool[sync_str], bins=bins, alpha=0.5, color=paler_color, label='Fast Tapping Group (N=2)')
    ax[i].set_title(sync_str)
    
    ax[i].set_xlim([0,1.5])
    ax[i].legend()

#ax[0].legend()
ax[i].set_xlabel('normalized ITI')
ax[1].set_ylabel('tap count')
 
   
    #ax[i].legend()
plt.tight_layout()
#fig.suptitle(f'histogram of nITIs')

################## SAVE FILE ???? ###########################
fig.savefig('./plots/nITI distributions histograms.png', dpi=120)

# fig.savefig(os.path.join('/Users/nolanlem/Documents/kura/swarmgen/old-exp-plots', f'nITI distributions histograms.eps'))


   
#%%######### beat optimization algo 

##### 2. NB: NO BEAT BINNING.......DEFAULT!!!!!!
beat_binning_flag = 'w NO beat binning'
for sync_cond, sync_str in zip(allconditions, sync_strs):
    for sync_cond_version in sync_cond:
        print('working on', sync_cond_version)
      
        sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
                    
        binned_subject_taps[sync_cond_version] = []
        
        for person in subject:
            try:
                tap_resps = subject_resps[person][sync_cond_version] # get subject taps per stim 
                #binned_taps = binBeats(tap_resps, sndbeatbin)
                # beat binning for ITI or no? 
                #binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                #binned_subject_taps[sync_cond_version].append(binned_taps) # save subject's binned_taps per stim
    
                # accumulate subject taps per sync_cond
                subject_sync_cond_taps[person][sync_str].append(tap_resps)
                
                # get normalized ITI vector and add it to the subject array
                normalized_tap_iti = list(np.diff(tap_resps)/centerperiods[sync_cond_version])
                # if longer than 19 beats, take last 19 beats 
                # if len(normalized_tap_iti) > 19:
                #     normalized_tap_iti = normalized_tap_iti[len(normalized_tap_iti)-19:]
                subject_sync_cond_itis[person][sync_str].append(normalized_tap_iti)
                #print(subject_sync_cond_itis[person][sync_str][v])
    
            except KeyError:
                #print('did not tap to ', sync_cond_version)
                pass
###############################
###############################################



#%% ###################################################################
##############  ITI TIME COURSE PLOTS #######################
###################################################################


####### ITI SLICING AND TAKE MEAN/STD

beatsegments = [(0,3), (3,6), (6,9)]

# dictionaries to hold mx, sx, and mx/sx errors 
iti_segment_mx, iti_segment_sx= {},{}
iti_mx, iti_sx = {}, {}
iti_mx_error, iti_sx_error = {}, {}

# beat str array for csv file output 
beat_strs = [str(i) for i in range(len(beatsegments))]

beat_segment_dir = os.path.join('plots', f'{len(beatsegments)}-beat-segment')

if os.path.exists(beat_segment_dir) == False:
    os.mkdir(beat_segment_dir)

itis_dir = beat_segment_dir + '/ITIs/'
pcs_dir = beat_segment_dir + '/PCs/'
csvs_dir = beat_segment_dir + '/csvs/'

########## make all the directories #################
makeDir(beat_segment_dir)
makeDir(itis_dir) # make ITI subdir
makeDir(pcs_dir) # make PCs subdir 
makeDir(csvs_dir) # make csvs subdir 

# for PC directories, model and subjects subdirs and phases dir 
model_dir = pcs_dir + 'model/'
subject_dir = pcs_dir + 'subject/'
makeDir(model_dir)
makeDir(model_dir + 'phases/')
makeDir(subject_dir)
makeDir(subject_dir + 'phases/')

#####################################
# timestamp for csv file and cross referencing plot with csv 
now = datetime.datetime.now()
timestamp = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(now.minute)   

makeDir(beat_segment_dir + '/ITIs/' + 'subject-ITI-plots')


#%% ############### PLOT WITHIN SUBJECT ITI AVERAGES AGGREGATED ON SAME PLOT WITH FILTERED SUBJECTS

## intialize dictionaries 
mx_all_subject_iti_seg = {}
sx_all_subject_iti_seg = {}

for subcatstr in ['loyalists', 'converters', 'all']:
    mx_all_subject_iti_seg[subcatstr], sx_all_subject_iti_seg[subcatstr] = {},{}
    for sync_str in sync_conds:
        mx_all_subject_iti_seg[subcatstr][sync_str], sx_all_subject_iti_seg[subcatstr][sync_str] = [],[]


fig_loyal, ax_loyal = plt.subplots(nrows=2, ncols=2, figsize=(10,7), sharex=True, sharey='col')
fig_conv, ax_conv = plt.subplots(nrows=2, ncols=2, figsize=(10,7), sharex=True, sharey='row')

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex=True, sharey='row')

 
#figs = [fig_loyal, fig_conv, fig_all]
#axs = [ax_loyal, ax_conv, ax_all]

n = 0       
for subjectbatch, subcatstr in zip([loyalists[:-1], converters], ['Regular Group (N=8)', 'Fast Group (N=2)']):
    
    error_range = np.arange(len(beatsegments), dtype=np.float)

    for j, sync_str in enumerate(sync_conds):
        mx_sync_avgs, sx_sync_avgs = [], []
        for person in subjectbatch:
            sc = 0   
            person_iti = pd.DataFrame(subject_sync_cond_itis[person][sync_str])
            
            mx_subj_iti_seg, sx_subj_iti_seg = [],[]
            for beatseg, beat_str in zip(beatsegments, beat_strs):
                tap_col = person_iti.iloc[:,beatseg[0]:beatseg[1]].values
                mx = np.nanmean(tap_col)
                sx = np.nanstd(tap_col)
                mx_subj_iti_seg.append(mx)
                sx_subj_iti_seg.append(sx)

            mx_sync_avgs.append(mx_subj_iti_seg)
            sx_sync_avgs.append(sx_subj_iti_seg)
            ### plot individual subject ITI per beat segment
        
        mx_sync_avgs = np.array(mx_sync_avgs)
        version_str = 'exp 1'        
        mxs_avg = np.nanmean(mx_sync_avgs, axis=0)
        sxs_avg = np.nanmean(sx_sync_avgs, axis=0)
        
        # CALCULATE SEM, sigma/sqrt(sample_size) 
        mxs_error = np.nanstd(mx_sync_avgs, axis=0)/np.sqrt(len(mx_sync_avgs))
        sxs_error = np.nanstd(sx_sync_avgs, axis=0)/np.sqrt(len(sx_sync_avgs))
        
        negval = sx_subj_iti_seg - sxs_error
        asym_error = [[0,0],[0,0],[0,0]]
        
        for i, elem in enumerate(negval):
            if elem < 0:
                asym_error[i] = [sx_subj_iti_seg[i], sxs_error[i]] 
            else:
                asym_error[i] = [sxs_error[i], sxs_error[i]]
        
        asym_error = np.array(asym_error).T
        ax[0,n].errorbar(error_range, mx_subj_iti_seg, yerr=mxs_error, linewidth=0.9, alpha=1, label=sync_str, capsize=5, marker='o')
        ax[1,n].errorbar(error_range, sx_subj_iti_seg, yerr=asym_error, linewidth=0.9, alpha=1, label=sync_str, capsize=5, marker='o')                        
        
        error_range += 0.1       
        
        ax[0,n].set_ylim([0.1, 1.1]) # MX PLOTS Y RANGE
        ax[1,n].set_ylim([-0.01, 0.3]) # SX PLOTS Y RANGE

 
        ax[0,n].set_xticks(np.arange(len(beatsegments)))
        ax[1,n].set_xticks(np.arange(len(beatsegments)))
        
        ax[0,n].set_xticklabels([str(elem+1) for elem in range(len(beatsegments))])
        ax[1,n].set_xticklabels([str(elem+1) for elem in range(len(beatsegments))])
        
        # make y axis log scale
        # ax[0,n].set_yscale('log')
        # ax[1,n].set_yscale('log')
        
        # logtick labels 
        xlabs = [0.25, 0.5, 0.75, 1.]
        ax[0,n].set_yticks(xlabs) # for row
        ax[0,n].set_yticklabels(xlabs)  # labels       
        
        ylabs = [0, 0.05, 0.1, 0.15, 0.2]
        ax[1,n].set_yticks(ylabs) # for cols
        ax[1,n].set_yticklabels(ylabs) # labels
        #ax[1,n].set_yticks([0,0.25,0.5, 1.])

        ax[1,n].set_xlabel('Tap Section')
        ax[0,n].set_title(subcatstr)
        ax[1,n].set_title(subcatstr)

    ax[n,0].set_ylabel('normalized ITI')   
    ax[n,1].legend(loc='lower right', bbox_to_anchor=(1.4,1.05))

    n+=1
 
# ##save log or no-log?
# fig.savefig(itis_dir + 'nITIs-loyalist-converters-lin.png', dpi=120)
# fig.savefig(itis_dir + 'nITIs-loyalist-converters-lin.eps')
# non-log y axis 

fig.savefig(itis_dir + 'nITIs-loyalist-converters-linear.png',dpi=120)
fig.savefig(itis_dir + 'nITIs-loyalist-converters-linear.eps')


#%%
#####################################################################
################### PHASE COHERENCE ANALYSIS################### ##############################################################
#####################################################################


# beat binning helper functions

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)

            if num_taps_in_bin > 1:   
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) 
            if num_taps_in_bin == 0:
                binnedtaps.append(np.nan)
            if num_taps_in_bin == 1:
                binnedtaps.append(taps[i][0])
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim

#%%################################################################################################################  
############################ PHASE COHERENCE SWARM PLOTS #########################
################################################################################################################

###### NEED TO SET IF YOU ARE MAKING CONVERTS or LOYALIST by 
### changing intialization of usable subjects  
### have to save them one by one  

subjectbatch = np.copy(loyalists)
subjectbatchstr = 'loyalists' # aka 'regular' 

# subjectbatch = np.copy(converters)
# subjectbatchstr = 'converters'  # aka 'fast'   
   

sns.set()
sns.set_palette(sns.color_palette("Paired"))


# lots of dictionaries (not using most)
binned_taps_per_cond = {}
subject_binned_taps_per_cond = {}
subject_binned_taps_per_stim = {}
all_osc_binned_taps_per_stim = {}
all_subject_binned_taps_per_stim = {}
all_subject_binned_taps_per_cond = {}
all_subject_taps_per_cond = {}


# subject and stim polar plot
# all aggregate, no beat sections
fig_all, ax_all = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.05, 'right':0.9}, 
                            figsize=(12,4), 
                            sharex=True)

# axis formatting
for ax_a in ax_all.flat:   
    ax_a.set_thetagrids([])
    ax_a.set_yticklabels([])
    ax_a.set_axisbelow(True)
    ax_a.grid(linewidth=0.1, alpha=1.0)  

sns.set(style='darkgrid')

the_osc_phases, osc_phases_cond = {}, {}

random_color = np.random.random(4000)

# right R vals to csv file 
R_csv = open(beat_segment_dir + '/PCs/R_csv.csv', 'w')
R_writer = csv.writer(R_csv)
R_writer.writerow(['coupling condition', 'beat section', 'R model', 'R subject', 'psi model', 'psi subject'])


osc_phases_cond, all_subject_binned_taps_per_cond  = {},{}


sc = 0

for sync_conds, sync_str in zip(allconditions, sync_strs):
    the_osc_phases = {}
    
    all_subject_binned_taps_per_stim= {}
    all_subject_taps_per_cond = {}
    
    osc_phases_cond[sync_str] = []
    all_subject_binned_taps_per_cond[sync_str] = []


    for sync_cond_version in sync_conds:
        #print('working on %r'%(sync_cond_version))
            
        osc_phases = {}
        stim_phases_sec = {}
        
        sndbeatbin = librosa.time_to_samples(sndbeatbins[sync_cond_version])
        y, _ = librosa.load('../stimuli/' + sync_cond_version + '.mp3')
        phases = np.load('./phases-zcs-exp-1/' + sync_cond_version + '.npy', allow_pickle=True)
        
        the_osc_phases[sync_cond_version] = []

        ################## GENERATIVE MODEL ##################################
        for p, osc in enumerate(phases):
            binned_zcs = binBeats(osc, sndbeatbin)
            binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)
            osc_phases[str(p)] = []
            
            for i in range(1, len(sndbeatbin)):
                zctobin = binned_zcs[i-1]
                binmin = sndbeatbin[i-1]
                binmax = sndbeatbin[i]
                bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                osc_phases[str(p)].append(float(bininterp(zctobin)))
            
            the_osc_phases[sync_cond_version].append(osc_phases[str(p)])
        
        osc_phases_cond[sync_str].extend(the_osc_phases[sync_cond_version])
        
        ################# SUBJECTS TAPS ###################################
        all_subject_binned_taps_per_stim[sync_cond_version] = []  
        
        for person in subjectbatch:
            try:
                taps = subject_resps[person][sync_cond_version]
                
                tap_iti = np.diff(taps)
                tap_mean = np.nanmean(tap_iti)
                tap_std = np.nanstd(tap_iti)
                tap_resps_algo = [tap for tap in tap_iti if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                tap_resps_secs = [taps[t] for t, tap in enumerate(tap_iti) if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                tap_resps_samps = librosa.time_to_samples(tap_resps_secs)
 
                
                binned_taps = binBeats(tap_resps_samps, sndbeatbin)
                binned_taps, avg_taps_per_bin = binTapsFromBeatWindow(binned_taps) 
                 
                subject_binned_taps_per_stim[person] = []
                
                for i in range(1, len(sndbeatbin)):
                    taptobin = binned_taps[i-1]
                    binmin = sndbeatbin[i-1]
                    binmax = sndbeatbin[i]
                    bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                    subject_binned_taps_per_stim[person].append(float(bininterp(taptobin)))
                    
                all_subject_binned_taps_per_stim[sync_cond_version].append(subject_binned_taps_per_stim[person])
                
                
            except:
                pass
            
                            
        all_subject_binned_taps_per_cond[sync_str].extend(all_subject_binned_taps_per_stim[sync_cond_version])        
            
    ############### take dataframes from subject, model per coupling cond ###################################
    df_model  = pd.DataFrame(osc_phases_cond[sync_str]) 
    df_subject = pd.DataFrame(all_subject_binned_taps_per_cond[sync_str])     # (470,19) = (5*tempo*2versions * 47 subjects) for coupling cond and timbre version                        
    
    agg_subj_taps = df_subject.values.flatten() 
    agg_model_taps = df_model.values.flatten()
    
    R_subject = np.abs(np.nanmean(np.exp(1j*agg_subj_taps)))
    psi_subject = np.angle(np.nanmean(np.exp(1j*agg_subj_taps)))          

    R_model = np.abs(np.nanmean(np.exp(1j*agg_model_taps)))
    psi_model = np.angle(np.nanmean(np.exp(1j*agg_model_taps)))            


    psi_subject_centered = psi_subject - psi_model
    agg_subj_taps -= psi_model
    agg_model_taps -= psi_model 
    psi_model_centered = 0  

    agg_subj_taps = agg_subj_taps[~np.isnan(agg_subj_taps)]
    agg_model_taps = agg_model_taps[~np.isnan(agg_model_taps)]

    p_subj = rayleightest(agg_subj_taps%(2*np.pi))
    p_stim = rayleightest(agg_model_taps%(2*np.pi))

    randomnoise_subject = np.random.random(len(agg_subj_taps))*0.3
    randomnoise_model = np.random.random(len(agg_model_taps))*0.3
    
    print(f'{sync_str} p_subj : {p_subj}')
    print(f'\t p : {p_stim}')
    # COMBINED SUBJECT + MODEL BUT NO BEAT SECTIONS
    ax_all[sc].plot(np.arange(2), np.arange(2), alpha=0, color='white') # this is a phantom line, for some reason there's a bug with the arrow in this polar plot so just make it transparant and length 1            
    ax_all[sc].scatter(agg_subj_taps, 0.7-randomnoise_subject, s=20, alpha=0.2, c='steelblue', marker='.', edgecolors='none', zorder=0)
    
   
    
    ax_all[sc].arrow(0, 0.0, psi_subject_centered, R_subject, color='darkblue', linewidth=0.9, zorder=2, label='subject taps')            
    ax_all[sc].scatter(agg_model_taps, 1-randomnoise_model, s=20, alpha=0.1, c='firebrick', marker='.', edgecolors='none', zorder=0)
    ax_all[sc].arrow(0, 0.0, psi_model_centered, R_model, color='darkred', linewidth=0.9, zorder=1,label='stimulus onsets')
    # if sync_str == 'strong':
    #     ax_all[sc].legend()
    ax_all[sc].set_title(sync_str)
    
            
    sc += 1

#R_csv.close()

#for non-beat section aggregate tap distribtuion 
ax_all[2].legend(bbox_to_anchor=(1., 1.05), loc='upper left')
fig_all.suptitle(subjectbatchstr + " phase coherence plots")

########################################################
###### SAVE figure!!??? ######

fig_all.savefig(f'./plots/pc_{subjectbatchstr}.png', dpi=150)
fig_all.savefig(f'./plots/pc_{subjectbatchstr}.eps', dpi=150)

#%% plot phase coherence distributions for generative model all three beat sections collapsed 
import itertools
from astropy.stats import rayleightest, kuiper 

def rnd(num):
    return np.round(num,2)

def wrap(ph):
    phases = (ph + np.pi) % (2 * np.pi) - np.pi
    return phases

f = open(model_dir + './R_psi.csv', 'w')
writer = csv.writer(f)
writer.writerow(['coupling', 'R_m', 'R_s', 'psi_m', 'psi_s', 'rayleigh model', 'rayleigh subject'])

# fig, ax = plt.subplots(nrows=3, ncols=1, 
#                        gridspec_kw={'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
#                             figsize=(10,10), 
#                             sharex=True)

fig, ax = plt.subplots(nrows=3, ncols=1, subplot_kw=dict(polar=True), 
                       gridspec_kw={'wspace':0.2,'hspace':0.01,'top':0.9, 
                                    'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,10), 
                            sharex=True)

fig_h, ax_h = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(7,10))


for ax_, ax__, sync_str in zip(ax.flat, ax_h.flat, sync_strs):
    #ax_.set_thetagrids([])
    ax_.set_yticklabels([])
    ax_.set_axisbelow(True)
    ax_.grid(linewidth=0.1, alpha=1.0)
    ax_.set_ylabel(sync_str)

    ax__.set_yticklabels([])
    # ax__.set_axisbelow(True)
    ax__.set_ylabel(sync_str)    
    ax__.set_xticks(np.linspace(-np.pi,np.pi,7))
    xlabels = [r'$\pi$', r'-2$\pi /3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'2$\pi/3$', r'$\pi$']
    ax__.set_xticklabels(xlabels)
    

from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, norm
import matplotlib.mlab as mlab


    
for i,sync_str in enumerate(sync_strs):
    print(i)
    ######## MODEL ##########
    mtaps = np.array(list(itertools.chain(*osc_phases_cond[sync_str])))
    mtaps = mtaps[~np.isnan(mtaps)] # remove nans
    noise = np.random.random(len(mtaps))*0.3
    
    R_m = np.abs(np.nanmean(np.exp(1j*mtaps)))
    psi_m = np.angle(np.nanmean(np.exp(1j*mtaps)))

    mtaps = mtaps - psi_m 

    
    ax[i].scatter(mtaps, 1-noise, s=20, alpha=0.1, 
                  c='blue', marker='.', edgecolors='none', 
                  zorder=0)
    ax[i].arrow(0, 0.0, 0, R_m, color='black', linewidth=1, zorder=2)            

    phases = wrap(mtaps)
    nbins = 60
    n, bins, p = ax_h[i].hist(phases, bins=nbins, density=True, alpha=0.8)
    #ax_h[i].plot(np.arange(0, len(n)), n)

    (mu, sigma) = norm.fit(phases)
    y = stats.norm.pdf(bins, mu, sigma)
    ax_h[i].plot(bins, y, 'r--', linewidth=1 )


    ######### SUBJECTS #################
    staps = np.array(list(itertools.chain(*all_subject_binned_taps_per_cond[sync_str])))
    staps = staps[~np.isnan(staps)] # remove nans

    snoise = np.random.random(len(staps))*0.3

    staps = staps - psi_m    

    R_s = np.abs(np.nanmean(np.exp(1j*staps)))
    psi_s = np.angle(np.nanmean(np.exp(1j*staps)))
    

    ax[i].scatter(staps, 0.7-snoise, s=20, alpha=0.2, 
                  c='red', marker='.', edgecolors='none', 
                  zorder=0)
    ax[i].arrow(0, 0.0, psi_s, R_s, color='red', linewidth=1, zorder=2)            
    
    #write to csv
   
    raymod = rayleightest(mtaps)
    submod = rayleightest(staps)
    writer.writerow([sync_str, rnd(R_m), rnd(R_s), rnd(psi_m), rnd(psi_s), raymod, submod])
    

    
fig.savefig(model_dir + './model all beat sections collapsed.png', dpi=200)
fig_h.savefig(model_dir + './model histogram densities.png', dpi=160)
f.close()
#%%


