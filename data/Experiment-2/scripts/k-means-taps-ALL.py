#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:22:46 2021

@author: nolanlem
"""
import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

os.chdir('/Users/nolanlem/Documents/kura/swarmgen/')
import load_subjects_resps 


# load up the subject respsonses from the other script 
subject_resps, sndbeatbins = load_subjects_resps.load_subjects_resps()


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

def getTrigs(trigdir):
    zcs = np.load(trigdir, allow_pickle=True)
    zcs = [s2t(elem)-10 for elem in zcs]
    zcs = [elem[elem>0] for elem in zcs]
    return zcs

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins) +1)]
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

#%%



#%% NB: NEED TO HAVE subjects taps loaded up into subject_resps in environment
## can run first few code blocks of ITI-subjects-all-beats.py to get subj_resps
########## IMPORTANT: NEED TO SET TIMBRE STR TO OUTPUT CORRECT NT or T
batch = 'all'
#timbrestr = 't'
timbrestr = 'nt'
#timbrestr = 'all'

stimdirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']
rdir = './stim-no-timbre-5/stimuli_1/phases/pc/'
rootdir = os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/', timbrestr, batch)
ntdir = os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/csvs/')
tdir = os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/t/all/csvs/')

if timbrestr == 'nt':
    stimtimbredir = './stim-no-timbre-5/'
if timbrestr == 't':
    stimtimbredir = './stim-timbre-5/'
if timbrestr == 'all':    
    stims_nt_t = ['./stim-no-timbre-5/', './stim-timbre-5/']

    
rootdir = os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/', timbrestr, batch)
csvfile = os.path.join(rootdir, 'all-stim-tap-stats.csv')

if timbrestr == 'all':
    csvfile = os.path.join(ntdir, 'all-stats-4-7-' + 'nt-new.csv')
    df_nt = pd.read_csv(csvfile)
    csvfile = os.path.join(tdir, 'all-stats-4-7-' + 't-new.csv')
    df_t = pd.read_csv(csvfile)

df = pd.concat([df_nt, df_t])
#%% load up stimulus data 

stimuli = df['stim'].values
unique_stimuli = set(stimuli)

def wrapto0(psi_s):
    if psi_s < 0:
        psi_s = psi_s + np.pi
    if psi_s > 0:
        psi_s = psi_s - np.pi
    return psi_s
### get beat windows, get Model ZCs and derive Order Params, find center bpm from beat windows
R_model, psi_model, cbpm, zcs = {}, {}, {}, {}

for i, stim in enumerate(unique_stimuli):
    print(f'working on %r/%r'%(i, len(unique_stimuli)))
    
    stimsplit = stim.split('_')
    bname = '_'.join([stimsplit[0], stimsplit[2], stimsplit[3]]) + '.txt'
    for stimdir, stimint in zip(stimdirs, ['1','2','3','4']):
        if stimsplit[-1] == stimint:
            for thestimdir in stims_nt_t:
                rpath = os.path.join(thestimdir, stimdir, 'phases', 'pc', bname)
                cbpmpath = os.path.join(thestimdir, stimdir, 'phases','center-bpm', bname)
                bwindir = os.path.join(thestimdir, stimdir, 'phases','beat-windows', bname)
                trigsdir = os.path.join(thestimdir, stimdir, 'trigs', bname.split('.')[0] + '.npy')
                # get beat windows, convert to secs
                bwindows = np.loadtxt(bwindir)
                bwindows = s2t(bwindows)
                
                zcs_ = getTrigs(trigsdir)
                zcs[stim] = zcs_
                zcpool = mapZCs2Circle(zcs_, bwindows)
                
                R_m, psi_m = getCircMeanVector(zcpool)
                #Rvals = np.loadtxt(rpath)
                #R_model[stim] = np.nanmean(Rvals)
                R_model[stim] = R_m
                psi_model[stim] = psi_m #(psi_m + np.pi)%(2*np.pi) - np.pi # to push center of beat to 0 deg for plotting purposes 
    
                # calculate avg per because saved txt in center-bpm is wrong 
                biti = np.diff(bwindows)/2
                bcents = bwindows[:-1] + biti
                avgbpm = np.mean(np.diff(bcents))
                #cbpm[stim] = np.loadtxt(cbpmpath)
                cbpm[stim] = avgbpm        
#%%
# get phase coherence from subject taps 
subj_normiti, subj_normstd, subj_nPVI = [],[],[]
coupling, model_mxiti, model_sxiti = [], [], []
timbre, version, tempo, R_subj, psi_subj = [], [], [], [], []
R_mod, psi_mod, zerocrossings = [],[],[]

for i in range(df.shape[0]):
    subj = df.iloc[i, 1]    # the subject name in .csv format
    thestim = df.iloc[i, 2]     # the stimulus
    tresps = subject_resps[subj][thestim] # tap responses from subject
    if tresps[0] == '':
        tresps = np.array([np.nan])
    subjiti = np.diff(tresps) # iti of tap responses
    normavgiti = np.nanmean(subjiti/cbpm[thestim]) # NPVI for subject
    stdavgiti = np.nanstd(subjiti/cbpm[thestim]) # STD of ITI for subject 
 
    # get Phase Coherence for Subject Taps
    bwindows = returnBB(thestim)
    binnedbeats = binBeats(tresps, bwindows)
    bb, _ = binTapsFromBeatWindow(binnedbeats)
    cmaptaps = mapTaps2Circle(bb, bwindows)   
    R_s, psi_s = getCircMeanVector(cmaptaps)
    #psi_s = psi_s
    #psi_s = (psi_s + np.pi)%(2*np.pi) - np.pi # to push center of beat to 0 deg for plotting purposes
    
    # get norm iti and norm std for stimuli in gen model 
    bwindowiti = np.diff(bwindows)
    normbwindowiti = np.mean(bwindowiti/cbpm[thestim])
    stdbwindowiti = np.std(bwindowiti/cbpm[thestim])
    model_mxiti.append(normbwindowiti)
    model_sxiti.append(stdbwindowiti)
    
    zerocrossings.append(zcs[thestim])
    
    
    # normalized Pairwise-Variablity index (nPVI)
    nPVI = []
    for n in range(1, len(subjiti)):
        iti_diff = subjiti[n] - subjiti[n-1]
        iti_diff = np.abs(iti_diff/(subjiti[n] + subjiti[n-1])/2)
        nPVI.append(iti_diff)
    nPVI = np.array(nPVI)/(len(subjiti)-1)
    nPVI = 100*np.nansum(nPVI)
    #print(thestim, nPVI)
        
    subj_normiti.append(normavgiti)
    subj_normstd.append(stdavgiti)
    subj_nPVI.append(nPVI)
    coupling.append(thestim.split('_')[0])
    timbre.append(thestim.split('_')[1])
    version.append(thestim.split('_')[-1])
    tempo.append(thestim.split('_')[2])
    R_subj.append(R_s)
    psi_subj.append(psi_s)
    R_mod.append(R_model[thestim]) # this just orders it correctly 
    psi_mod.append(psi_model[thestim])
  
# add |R|, center bpm, nPVI, ITI, STD column to csv using pandas df
stimuli = df['stim'].values

R_ordered, cntbpm = [], []

[R_ordered.append(R_model[stim]) for stim in stimuli]
[cntbpm.append(float(per2bpm(cbpm[stim]))) for stim in stimuli]

df['R model'] = R_mod
df['R subject'] = R_subj
df['psi model'] = psi_mod
df['psi subject'] = psi_subj
df['center bpm'] = cntbpm
df['norm iti subj'] = subj_normiti
df['norm std subj'] = subj_normstd
df['nPVI'] = subj_nPVI
df['coupling'] = coupling
df['timbre'] = timbre
df['version'] = version
df['tempo'] = tempo
df['norm iti model'] = model_mxiti
df['norm std model'] = model_sxiti


### add 'taps', 'subject norm itis' (per beat), 'experiment version', 'beat window locations'
# let's look at iti plots of just the loyalists 
from ast import literal_eval
def removeStrFormatting(str_array):
    for str_arr in str_array:
        str_arr = str_arr[1:-1] # remove "'[" and "]'"
        str_arr = str.split(str_arr, ',') # split strings
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
        #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr

def getExpVersion(data):
    sp = data.split('_')[1]
    sp = sp.split('-')[1]
    return sp
def t2s(a):
    return librosa.time_to_samples(a)
def s2t(a):
    return librosa.samples_to_time(a)
def per2bpm(per):
    return np.round(60./(per),1)
def bpm2per(bpm):
    return np.round(1./(bpm/60),4)
def Hz2bpm(hz):
    return np.round(60.*hz, 2)

numbeats = 19
taps, expversions, subjectnormitis, beatbins = [], [], [], []
for substim in df.loc[:, ['subject', 'stim', 'center bpm']].values:
    person, stim, cbpm = substim
    expversion = getExpVersion(person)
    centerperiod = 1/(cbpm/60.)
    print(person, stim, expversion, centerperiod)
    expversions.append(expversion)
    thetaps = subject_resps[person][stim]
    if thetaps[0] == '':
        thetaps = np.array([np.nan])
    taps.append(thetaps)
    subjnormiti = np.diff(thetaps)/centerperiod
    subjectnormitis.append(subjnormiti)
    
    beatbins.append(s2t(sndbeatbins[stim]))
df['taps'] = taps
df['subjects norm itis'] = subjectnormitis
df['experiment version'] = expversions
df['beat windows'] = beatbins

###################################
## SAVE TO CSV, PKL? 
#df.to_pickle(rootdir + '/subject-stats-' + timbrestr + '-rev.pkl')
#df.to_csv(rootdir + '/subject-stats-' + timbrestr + '-rev.csv')
#%% sanity check 
y, _ = librosa.load('./allstims/' + thestim + '.wav')
plt.plot(y)
plt.vlines(t2s(bwindows), -1, 1, color='red')
plt.vlines(t2s(tresps), -1.5, -0.5, color='blue')
#%%
### NB: need to run this until you get 0 is all strong, loyalists then are at 1, and
### converters goto group 2. can verify with plot colors 
#### K MEANS CLUSTERING 
from sklearn.cluster import KMeans

print('doing k-means....')
def getKMeans(x, y, nclusters=2, xname='x', yname='y'):
    plt.figure(figsize=(8,8))
    
    kmeans = KMeans(n_clusters= nclusters)
    points = np.array([x, y]).T
    #breakpoint()
    lendata = float(points.shape[0])  
    # fit kmeans object to data
    kmeans.fit(points)
    y_km = kmeans.fit_predict(points)
    klen = []
    
    # order clusters from highest to lowest
    ordarr = []
    for k in range(nclusters):
        count = np.count_nonzero(y_km == k)
        ordarr.append((k,count))
    clusters_ = sorted(ordarr, key=lambda tup: tup[1], reverse=True)
    
    ###### switch group idx 2 with 1 and vv, hacky.. .may be different for timbre case
    ## also switch idx 0 with 1
    ord_clusters = [tup[0] for tup in clusters_]
    #breakpoint()
    tmp_idx_1 = ord_clusters[1]
    #tmp_idx_0 = ord_clusters[]
    ord_clusters[1] = int(ord_clusters[2])
    ord_clusters[2] = tmp_idx_1
    
    tmp_idx_1 = ord_clusters[0]
    ord_clusters[0] = int(ord_clusters[1])
    ord_clusters[1] = tmp_idx_1
    #breakpoint()
    ################
    # plot the points on a scatter plot
    for n,k in enumerate(ord_clusters):
        #print(n,k)
        if n == 4: # force group 5 to be cyan from color palette hacky
            print('here')
            plt.scatter(points[y_km == k,0], points[y_km == k,1], s=10, alpha=0.5, label=str(n+1), color=sns.color_palette()[9])
        else:
            plt.scatter(points[y_km == k,0], points[y_km == k,1], s=10, alpha=0.5, label=str(n+1))

        klen.append(len(points[y_km == k,0]))
        #print(f'category {k} had %r datapoints which was %r of distribution' %(len(points[y_km==k,0]), np.round(points[y_km==k,0]/lendata,2)))
        
    plt.xlabel(f'{xname}')
    plt.ylabel(f'{yname}')
    plt.hlines(1, 0, 1, linestyle='--', color='black', linewidth=0.6)
    plt.legend(title='Tapping Cluster')
    plt.tight_layout()
    #plt.title(timbrestr + ' subj norm ITI vs R mod K means cluster')
    plt.gca().set_yscale('log')
    yticks = [0.5, 1, 1.5, 2]
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([str(elem) for elem in yticks])
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


    #### SAVE FIG?#####
    #plt.savefig(os.path.join(rootdir, 'k-means-plots','new', f'K-mean-{nclusters}-clusters.eps'))
    plt.savefig(os.path.join(rootdir,'k-means-plots','new',  f'K-mean-{nclusters}-clusters.png'),dpi=120)
    
    #finalsdir = '/Users/nolanlem/Documents/kura/swarmgen/nt-paper/images/in-paper/'
    #plt.savefig(os.path.join(finalsdir, f'K-mean-{nclusters}-clusters.eps'))
    #plt.savefig(os.path.join(finalsdir,f'K-mean-{nclusters}-clusters.tif'),dpi=120)

    return points, y_km

#%%
### can start here 
df = pd.read_csv('../final-tapping-scripts/timbre-paper/df-bb-taps-3-2.csv')  

# remove .csvs
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]      

#%% ###### 
######## do k=5 k-means clustering  #############

timbre = 'n'
Rmodel = df[df['timbre'] == timbre]['R model'].values
normiti = df[df['timbre'] == timbre]['norm iti subj'].values
mx = np.nanmean(normiti)
normiti = np.nan_to_num(normiti, nan=mx)

numclusters = 5
points, y_km = getKMeans(Rmodel, normiti, nclusters=numclusters, xname='Stimulus Phase Coherence', yname='nITI')
#  see who's in the groups and separate 
dfgroups = {}
sortedkgroups = {}

# get group 0 and 3 rows in df
for i in range(numclusters):
    dfgroups[str(i)] = df.loc[df['norm iti subj'].isin(points[y_km==i, 1])]

n = 0
for k in sorted(dfgroups, key=lambda k: dfgroups[k].shape[0], reverse=True):
    sortedkgroups[str(n)] = dfgroups[k]
    n+=1 
    
# add k means group col to df
for i in [str(elem) for elem in range(numclusters)]:
    print(f'working on {i}')
    for person in sortedkgroups[i]['subject'].values:
        idxs = sortedkgroups[i][sortedkgroups[i]['subject'] == person].index
        for idx in idxs:
            df.loc[idx, 'k group'] = i 
            

#plt.savefig('/Users/nolanlem/Documents/kura/final-tapping-scripts/timbre-paper/new-saved-plots/k-means-50.png', dpi=150)
            
#%%

dense, sparse = [], []
dfslice_dense = df[(df['k group'] == '2') & (df['dispersion'] < 0.1065)].values

np.mean(df[(df['k group'] == '2') & (df['dispersion'] > 0.1065) & (df['norm iti subj'] < 0.5)]['mean raw iti'].values)

## sparse 
# regular: 366, 667, 419  
# hybrid: 333, 517, 370, 
# fast: 280, 377, 301                      

# dense: 
# regular: 298, 617, 430
# hybrid: 437, 624, 458
# fast: 209, 359, 232
#%%
cutoff = 1.
print(f'@ {cutoff}')
for sc in ['dense', 'sparse']:
    print('')
    mx = 0 

    for subcat in ['regular', 'hybrid', 'fast']:
  
        if sc == 'dense':
            mx = df[(df['subject cat'] == subcat) & (df['k group'] == '2') & (df['dispersion'] < 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].mean()
            cnt = df[(df['subject cat'] == subcat) & (df['k group'] == '2') & (df['dispersion'] < 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].count()
        else:
            mx = df[(df['subject cat'] == subcat) & (df['k group'] == '2') & (df['dispersion'] > 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].mean()
            cnt = df[(df['subject cat'] == subcat) & (df['k group'] == '2') & (df['dispersion'] > 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].count()            
        print(f'{sc}, ({cnt}) {subcat}: {mx}')
            
    
    if sc == 'dense':
        mx = df[(df['k group'] == '2') & (df['dispersion'] < 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].mean()
        cnt = df[(df['k group'] == '2') & (df['dispersion'] < 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].count()
    else:
        mx = df[(df['k group'] == '2') & (df['dispersion'] > 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].mean()
        cnt = df[(df['k group'] == '2') & (df['dispersion'] > 0.1065) & (df['mean raw iti'] < cutoff)]['mean raw iti'].count()
    print(f'{sc}, ({cnt}) all mx: {mx}')
#%% ########## 
####### Just plot the imitators on the k-means plot


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
imitators = ['PARTICIPANT_kura-A1_2020-10-05_11h28.35.951.csv','PARTICIPANT_kura-B1_2020-10-05_10h39.48.518.csv']
densesubs = ['PARTICIPANT_kura-A1_2020-10-05_06h11.28.743.csv', 'PARTICIPANT_kura-B2_2020-09-07_10h04.17.709.csv','PARTICIPANT_kura-A2_2020-09-07_10h52.13.829.csv', 'PARTICIPANT_kura-A2_2020-09-16_12h19.57.327.csv', 'PARTICIPANT_kura-A2_2020-09-16_15h15.27.472.csv', 'PARTICIPANT_kura-B1_2020-09-16_15h23.50.906.csv', 'PARTICIPANT_kura-B2_2020-09-07_08h44.56.737.csv']

# select imitators data
Rmodelimit = df[df['subject'].isin(imitators)]['R model'].values
normiti = df[df['subject'].isin(imitators)]['norm iti subj'].values
mx = np.nanmean(normiti)
normiti_imit = np.nan_to_num(normiti, nan=mx)
# select dense 
Rmodeldense = df[df['subject'].isin(densesubs)]['R model'].values
normiti = df[df['subject'].isin(densesubs)]['norm iti subj'].values
mx = np.nanmean(normiti)
normiti_dense = np.nan_to_num(normiti, nan=mx)




ax.scatter(x=Rmodelimit, y=normiti_imit, s=10, alpha=0.5, label=f'imitators (N={len(imitators)})', color='blue')
ax.scatter(x=Rmodeldense, y=normiti_dense, s=10, alpha=0.5, label=f'dense (N={len(densesubs)})', color='red')

plt.xlabel(f'Stimulus Phase Coherence')
plt.ylabel(f'nITI')
plt.hlines(1, 0, 1, linestyle='--', color='black', linewidth=0.6)
plt.legend(title='Tapping Cluster')
plt.tight_layout()
#plt.title(timbrestr + ' subj norm ITI vs R mod K means cluster')
plt.gca().set_yscale('log')
yticks = [0.5, 1, 1.5, 2]
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels([str(elem) for elem in yticks])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


plt.savefig(os.path.join(rootdir,'k-means-plots','new',  f'k-means-imitators-dense.png'),dpi=120)
#%%

regulars = [
'PARTICIPANT_kura-A2_2020-11-25_15h04.36.125.csv',
'PARTICIPANT_kura-A2_2020-11-05_09h19.11.832.csv',
'PARTICIPANT_kura-A1_2020-10-04_12h32.21.589.csv',
'PARTICIPANT_kura-A1_2020-10-01_10h17.21.641.csv',
'PARTICIPANT_kura-B2_2020-09-10_18h03.12.198.csv',
'PARTICIPANT_kura-B2_2020-09-25_19h37.04.527.csv',
'PARTICIPANT_kura-A1_2020-10-05_06h37.13.057.csv',
'PARTICIPANT_kura-A1_2020-10-03_10h29.30.435.csv',
'PARTICIPANT_kura-A1_2020-10-05_10h32.11.341.csv',
'PARTICIPANT_kura-B1_2020-10-05_07h49.31.729.csv',
'PARTICIPANT_kura-B2_2020-09-07_12h08.36.208.csv',
'PARTICIPANT_kura-A1_2020-10-05_09h41.04.130.csv',
'PARTICIPANT_kura-A2_2020-11-04_17h32.29.185.csv',
'PARTICIPANT_kura-A2_2020-09-16_10h16.41.575.csv',
'PARTICIPANT_kura-A2_2020-09-16_17h20.10.296.csv',
'PARTICIPANT_kura-B2_2020-09-07_21h13.50.539.csv'
]

hybrids = [
'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv',
'PARTICIPANT_kura-B2_2020-09-14_22h18.47.480.csv',
'PARTICIPANT_kura-A2_2020-09-07_10h21.43.606.csv',
'PARTICIPANT_kura-B2_2020-09-13_16h50.25.586.csv',
'PARTICIPANT_kura-B1_2020-09-10_21h13.09.622.csv',
'PARTICIPANT_kura-B1_2020-09-07_11h57.44.574.csv',
'PARTICIPANT_kura-A2_2020-09-18_08h50.45.006.csv',
'PARTICIPANT_kura-B2_2020-09-07_08h44.56.737.csv',
'PARTICIPANT_kura-A2_2020-09-16_12h19.57.327.csv',
'PARTICIPANT_kura-A1_2020-10-05_06h11.14.448.csv',
'PARTICIPANT_kura-B1_2020-10-05_08h31.10.554.csv',
'PARTICIPANT_kura-A2_2020-08-28_09h24.01.653.csv',
'PARTICIPANT_kura-A2_2020-09-07_11h08.21.418.csv',
'PARTICIPANT_kura-B1_2020-09-18_23h19.34.883.csv',
'PARTICIPANT_kura-B2_2020-09-16_12h17.20.371.csv',
'PARTICIPANT_kura-B2_2020-08-13_23h21.57.157.csv',
'PARTICIPANT_kura-A2_2020-11-06_20h46.36.344.csv',
'PARTICIPANT_kura-B2_2020-09-21_10h01.20.704.csv',
'PARTICIPANT_kura-B2_2020-09-16_13h54.10.653.csv',
'aa_kura-A1_2020-08-08_23h03.30.895.csv',
'PARTICIPANT_kura-A2_2020-09-16_12h21.37.513.csv',
'PARTICIPANT_kura-B1_2020-10-05_07h34.10.665.csv',
'PARTICIPANT_kura-A1_2020-10-01_08h10.36.781.csv',
'PARTICIPANT_kura-A2_2020-09-16_13h15.20.738.csv',
'PARTICIPANT_kura-B1_2020-10-05_09h20.02.892.csv'
]

fasts = [
'PARTICIPANT_kura-B1_2020-10-05_08h33.11.406.csv',
'PARTICIPANT_kura-A1_2020-10-05_06h11.28.743.csv',
'PARTICIPANT_kura-B2_2020-09-07_10h04.17.709.csv',
'PARTICIPANT_kura-B1_2020-09-16_15h23.50.906.csv',
'PARTICIPANT_kura-A2_2020-09-07_10h52.13.829.csv',
'PARTICIPANT_kura-B1_2020-10-05_10h39.48.518.csv',
'PARTICIPANT_kura-A2_2020-09-16_15h15.27.472.csv',
'PARTICIPANT_kura-A1_2020-10-05_11h28.35.951.csv'
]


#%%### NB: loyalists need to be group 0 and group 1 and convert/nc need to be 2!!!!!! 
#### keep running last block until that shows up in plot

#df.to_csv(rootdir + '/k-stats-4-7-' + timbrestr + '.csv') 

#df = pd.read_pickle('./mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/csvs/all-stats-4-7-nt.pkl')
#df = pd.read_pickle('./mturk-csv/usable-batch-12-7/subject-raster-plots/t/all/csvs/all-stats-4-7-t.pkl')
df = pd.read_csv('./mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/csvs/all-stats-4-7-tnt.csv')
#%%
import csv
fi = open('csvs/' + timbrestr + '-subject-categories.csv', 'w')
writer = csv.writer(fi)
writer.writerow(['subject', 'coupling', '0', '1', '2', '3', '4', 'category', 'timbre'])


unique_subjects = list(set(df['subject']))

# accum subjects who responded to none, weak, and medium stim as a converter
theloyalists, theconverts, noneconverts = [], [], []

#fig, ax = plt.subplots(ncols=1, nrows=50, figsize=(16,45), sharey=True, sharex=True)
fig = plt.figure(figsize=(15,15))
xrange = np.arange(4)
width = 0.35

numsubjects = len(list(set(df[df['k group'] <= '3']['subject'])))

for i, person in enumerate(list(set(df.subject))):
#for i, person in enumerate(['PARTICIPANT_kura-B1_2020-10-05_07h49.31.729.csv']):
    couplinglisttmp = []
   
    print(f'{person}')
    cpgrp = []
    for cp in ['none', 'weak', 'medium', 'strong']:
        for n in [0, 1, 2, 3, 4]:
            subset = df[(df['subject'] == person) & (df['coupling'] == cp) & (df['k group'] == str(n))]
            numpercp = subset.shape[0]
            #print(f' {cp} group {n} = {numpercp} ')
            cpgrp.append(numpercp)
     
    # this is hacky but grp none: group 0,1,2,3, weak: 0,1,2,3, etc.
    n0, n1, nc, n3, n4, w0, w1, wc, w3, w4, m0, m1, mc, m3, m4, s0, s1, sc, s3, s4 = list(cpgrp) # unzip
    # loyalists are group 0 and group 1 combined
    # nc, wc, mc, sc are converters 
    nl = n0+n1
    wl = w0+w1
    ml = m0+m1
    sl = s0+s1
    
    n34 = n3 + n4
    w34 = w3 + w4
    m34 = m3 + m4
    s34 = s3 + s4
    
    ax = fig.add_subplot(8,7,i+1)
    ax.bar(np.linspace(0,3,4), [nl, wl, ml, sl], width=2*width/3, label='loyalists')
    ax.bar(np.linspace(0.3,3.3,4), [nc, wc, mc, sc], width=2*width/3, label='converters')
    ax.bar(np.linspace(0.6, 3.6, 4), [n34, w34, m34, s34], width=2*width/3, label='neither')
    
    ax.set_xticks(xrange)
    ax.set_xticklabels(['n', 'w', 'm', 's'])
    ax.set_title(person.split('.')[-2])
    ax.set_ylim([0,10])
    

    #writer.writerow([person, nl, nc, wl, wc, ml, mc, sl, sc])

    # save who are converters 
    # e.g. (num stim that they converted to was > nums stims they stayed loyal
    # for none, weak, medium coupling cond)
    if nc > nl:
        if wc > wl and mc >= ml:
            tmpcat = 'convert'
            theconverts.append(person)
        if wc < wl or mc < ml:
            tmpcat = 'noneconvert'
            noneconverts.append(person)
    else:
        tmpcat = 'loyalist'
        theloyalists.append(person)

    writer.writerow([person, 'none', n0, n1, nc, n3, n4, tmpcat, timbrestr])


ax = fig.add_subplot(8,7,i+2)
ax.bar([0],[0], label='loyalists')
ax.bar([0],[0], label='converter')
ax.bar([0],[0], label='neither')


ax.legend()
plt.tight_layout()    
print('\n')
print(f'in total there were {len(theloyalists)} / {numsubjects} subjects who stayed loyal the whole time for each coupling cond')
print(f'in total there were {len(theconverts)} / {numsubjects} subjects who converted more than staying loyal for none, weak, and medium coupling condition')   
print(f'in total there were {len(noneconverts)} / {numsubjects} subjects who converted only on the none coupling cond')   
'''
no-timbre 
    in total there were 17 / 52 subjects who stayed loyal the whole time for each coupling cond
    in total there were 9 / 52 subjects who converted more than staying loyal for none, weak, and medium coupling condition
    in total there were 26 / 52 subjects who converted only on the none coupling cond
timbre: 
    in total there were 29 / 52 subjects who stayed loyal the whole time for each coupling cond
    in total there were 9 / 52 subjects who converted more than staying loyal for none, weak, and medium coupling condition
    in total there were 14 / 52 subjects who converted only on the none coupling cond
'''

plt.title(timbrestr + ' loyalist, converter, neither bar chart per subject')

plt.savefig(os.path.join(rootdir, 'loyalist vs converts bar plot.png'), dpi=150)
        
bcols = np.empty((df.shape[0], numbeats,))
bcols[:] = np.nan

for i, tapiti in enumerate(subjectnormitis):
    if len(tapiti) <= numbeats:
        for b in range(len(tapiti)):
            bcols[i, b] = tapiti[b]
    else:
        for b in range(numbeats):
            bcols[i, b] = tapiti[b]

bnums = [str(elem) for elem in range(numbeats)]
    
for i, b in enumerate(bnums):
    df[b] = bcols[:, i]


loyaliststr = [] 
#i = 0
for index, row in df.iterrows():
    #print(i)
    if row['subject'] in theloyalists:
        loyaliststr.append('loyalist')
    if row['subject'] in noneconverts:
        loyaliststr.append('noneconvert')
    if row['subject'] in theconverts:
        loyaliststr.append('convert')

    #i+=1

df['subject cat'] = loyaliststr

df.to_pickle(rootdir + '/csvs/all-stats-4-7-' + timbrestr + '.pkl')
df.to_csv(rootdir + '/csvs/all-stats-4-7-' + timbrestr + '.csv')
#df.to_pickle(os.path.join(rootdir, 'csvs', 'all-stats-3-16.pkl'))
        

#%%


sns.pairplot(data=df.iloc[:, :-19], hue='coupling')

#%% norm iti subj vs. norm iti mod 
#sns.scatterplot(data=df, x='norm iti model', y='norm iti subj', hue='coupling', ax=ax)
df['k group'] = df['k group'].replace(np.nan, 4) # all np.nans move to group 4 which is dont care
g = sns.FacetGrid(df[df['k group'] < 3], col="k group", hue="coupling")
g.map(sns.scatterplot, "norm iti subj", "norm std subj", alpha=.7)
#%%
for grp in ['0', '1', '2']:
    plt.subplot(3,1,int(grp)+1)
    g = sns.lmplot(data=df[df['subject'].isin(theloyalists)], x='norm iti model', y='norm iti subj', 
                    hue='coupling', line_kws={'color':'purple', 'linestyle': '--'},
                    ax=ax[int(grp)])
    # sns.regplot(data=df, x='norm iti model', y='norm iti subj', ax=ax, 
    #                line_kws={'linewidth': 0.5, 'color':'red', 'linestyle':'--', 'alpha': 0.3})
    # #ax.plot(np.arange(0,3))
    #ax.set_xlim([0.9, 1.2])
    #ax.hlines(1, ax.get_xlim()[0], ax.get_xlim()[1], color='red', linewidth=0.5)
    plt.gca().hlines(1, 0.9, 1.1, color='black', linestyle='--')
    g.set(xlim=(0.9,1.1))

#%% norm iti subj vs. norm iti mod 
g = sns.jointplot(x=df['norm std subj'], y=df['norm std model'], hue=df['R subject'],
                  legend=False)
g.ax_marg_x.set_xlim(g.ax_marg_y.get_ylim())
#%%
from scipy import interpolate
bbins = np.arange(1, 11)
bbins = bbins + np.random.random(size=len(bbins))
xrange = np.arange(1, 11)
f = interpolate.interp1d(xrange, bbins)

freq = 2
period = 1/freq
xnew = np.arange(1, 10, period)
ynew = f(xnew)

plt.plot(xrange, bbins, '.', xnew, ynew, 'x')



#%%
bwind = np.loadtxt('./stim-no-timbre-5/stimuli_1/phases/beat-windows/medium_105_1.txt', delimiter=',')
bwind = s2t(bwind)
#%%
xrange = np.arange(0, len(bwind))
f = interpolate.interp1d(xrange, bwind)

freq = 2
period = 1/freq
xnew = np.arange(0, len(bwind), period)
xnew = xnew[xnew<19]
ynew = f(xnew)

plt.plot(xrange, bwind, '.', xnew, ynew, 'x')


#%%
taps = np.arange(1,11)
amt_noise = 0.5
taps = taps + np.random.uniform(low=-amt_noise, high=amt_noise, size=len(taps))

bw = np.arange(1,11)
s = []
for tap in taps:
    s.append(find_nearest(bw, tap))
print(s)
#%%


def find_nearest_diff(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    diff = np.abs(array[idx] - value)
    diff_sq = diff**2
    return diff_sq
# calculate the frequency normalized root-mean squared deviation (frmsd) 
# with isochronous freq grids 
# freq_res = how many steps between each Hz
def frmsd_iso(taps, shift_res=60, freq_res=30, freqlow=0.5, freqhigh=7):  
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

import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

#%% NB: don't need to run remove all stat plots from subject dir 

for person in subject:
    for fi in glob.glob(os.path.join(plotdir,person, '*.png')):
        print(f'removing {fi}')
        os.remove(fi)   

#%%#%%

def per2bpm(per):
    return np.round(60./(per),1)

def Hz2bpm(hz):
    return np.round(60.*hz, 2)

def reformatStim(thestim):
    thestim = thestim.split('_')
    newstimname = thestim[0] + '_' + thestim[-2] + '_' + thestim[-1] + '.txt'
    return newstimname

from scipy.signal import argrelextrema

def findPk(sig, numpks=1):
    pk_list = []
    for pk in range(numpks):
        height = max(sig[:])
        thepk = find_peaks(sig, height=height)
        #print(f'found peak at {thepk}') 
        if thepk[0].size != 0:
            pk_idx = thepk[0][0]
            pk_list.append(pk_idx)
            sig[pk_idx] = min(sig)
        else:
            pk_list.append(0)
    return np.array(pk_list)

# def findPk(sig, numpks=4):
#     height = max(sig)
#     pkidxs = argrelextrema(sig, np.greater)
#     pkidxs = pkidxs[0][:numpks]
#     return pkidxs

def plotStim(wf, taps, bb, bc, theax, thestim):
    theax.plot(y, linewidth=0.5, alpha=0.77) # plot stim wf
    theax.vlines(taps_s, -1, -1.7, color='red', linewidth=0.7) # plot taps
    theax.vlines(bbins, 0.5, -1.5, color='green', linewidth=0.7) # plot beat bins
    theax.vlines(bcenters, -1.5 , 1, color='orange', linewidth=0.7) # plot beat centers
    theax.set_title(thestim)
    
    wfrange = np.arange(0, len(y))
    xrange = np.arange(0, len(y), 22050*4) # every 4 seconds, show seconds on x label
    theax.set_xticks(xrange)
    xticklabs = [np.round(wfrange[elem]/22050.,2) for elem in xrange]
    theax.set_xticklabels(xticklabs, fontsize=8)
    
def plotGAT(tapgat, bcgat, freqbins, theax):
    # have to invert function to "find max" since we're plotting y ax inverted
    i_tapgat = np.array(tapgat)*-1 
    i_bcgat = np.array(bcgat)*-1 
    pkidx_t = findPk(i_tapgat, numpks=2)
    pkidx_bc = findPk(i_bcgat, numpks=4) 
    
    #theax.invert_yaxis()
    xrange = np.arange(0, len(freqbins),20)
    theax.set_xticks(xrange)
    # Hz? 
    #xticklabs = [int(Hz2bpm(freqbins_t[elem])) for elem in xrange]
    xticklabs = [np.round(freqbins[elem],2) for elem in xrange]
    
    theax.set_xticklabels(xticklabs, fontsize=10)
    theax.plot(tapgat, color='red', label='subject', linewidth=0.8)
    theax.plot(bcgat, color='blue', label='stimulus', linewidth=0.8)
    
    theax.set_ylabel('mean deviation (frmsd)') # frmsd = frequency root mean squared deviation
    
    # plot found peaks for both frmsds
    for pk in pkidx_t:
        theax.plot(pk, tapgat[pk], marker='x', color='darkred')
        theax.vlines(pk, tapgat[pk], max(tapgat), color='darkred', linewidth=0.8, linestyle='-')
        pk_hz = np.round(freqbins[pk],2)
        theax.annotate(str(pk_hz),(pk, tapgat[pk]), color='darkred', fontsize=8)
    
    for pk in pkidx_bc:
        theax.plot(pk, bcgat[pk], marker='x', color='darkblue')
        theax.vlines(pk, bcgat[pk], max(bcgat), color='darkblue', linewidth=0.8, linestyle='-')
        pk_hz = np.round(freqbins[pk],2)
        theax.annotate(str(pk_hz),(pk, bcgat[pk]), color='darkblue', fontsize=8)
    

def distributeTrigs(taps, srate=60):
    taps = np.array(taps)
    taps = taps*srate
    taps = [np.int(elem) for elem in taps]
    taptrigs = np.zeros(max(taps)+1)
    for trig in taps:
        taptrigs[trig] = 1
    
    win = signal.windows.hann(10) # 3000
    # convolve with narrow npdf
    filt_taps = signal.convolve(taptrigs, win, mode='same') / sum(win)
    return filt_taps

def ccorr(sig1, sig2):
    cc = np.correlate(sig1, sig2, mode='full')
    return cc[cc.size//2:]
#%%
#taps = np.linspace(1,21,10)
#stim = 'medium_n_119_3'
#stim = 'medium_n_72_3'
#stim = 'none_n_81_3'
#stim = 'none_n_72_4'
stim = 'strong_n_81_1'
stim = 'medium_n_72_3'
stim_txt = reformatStim(stim)
#person = 'PARTICIPANT_kura-A2_2020-09-07_10h52.13.829.csv'
person='PARTICIPANT_kura-B1_2020-10-05_08h33.11.406.csv' 

person='PARTICIPANT_kura-B1_2020-10-05_07h49.31.729.csv' #29

allntstims = all_timbre_conds[1]
sync_strs = ['none', 'weak','medium', 'strong']

a1b1stims = [[stim for stim in cond if stim.split('_')[-1] in ['1','2']] for cond in allntstims] # nt stims for A1, B1
a2b2stims = [[stim for stim in cond if stim.split('_')[-1] in ['3','4']] for cond in allntstims] # nt stims for A2, B2
# sort by tempo (increasing)
for batch1, batch2 in zip(a1b1stims, a2b2stims):
    batch1.sort(key=natural_keys)
    batch2.sort(key=natural_keys)

plotdir = '/Users/nolanlem/Documents/kura/swarmgen/mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/'

start_time = time.time()


for n,person in enumerate(imitators):

    print(f'working on {person} %r/%r'%(n, len(subject)))   
    v = person.split('_')[1].split('-')[1][-1]
    if v == '1':
        stimbatch = a1b1stims
    if v == '2':
        stimbatch = a2b2stims
    
    for i, sc in enumerate(stimbatch):
        fig, ax = plt.subplots(nrows=len(sc), ncols=5, figsize=(30,16), sharey='col', sharex='col')
        
        for ax_ in ax[:,0].flat:
            ax_.set_yticks([])
            
        ax[0,2].invert_yaxis() # invert y axis for GAT pulse analysis 
    
        for j, stim in enumerate(sc):
            print(f'\t working on {person} %r/%r in {stim} %r/%r %r/%r'%(n+1, len(subject), i+1, len(stimbatch), j+1, len(sc)))

            stim_txt = reformatStim(stim)
            
            bbfile = 'stim-no-timbre-5/stimuli_' + stim.split('_')[-1] + '/phases/beat-windows/' + stim_txt
            
            bbins = np.loadtxt(bbfile) # already in samples
            bbinsiti = np.diff(bbins)/2
            bcenters = bbins[:-1] + bbinsiti #make beat centers

            y, sr_ = librosa.load('./allstims/' + stim + '.wav')
                
            taps = subject_resps[person][stim]
            # if tap array is empty for some reason fill it with np.nan
            if len(taps) == 0:
                taps = np.array([0])
            taps_s = t2s(taps) # taps in secs to samples
            ##### PLOT STI ####
            plotStim(y, taps_s, bbins, bcenters, ax[j,0], stim)
            #####  PLOT HISTO for TAPS and STIM OVERLAYED####
            tapiti = np.diff(taps)
            bciti = np.diff(s2t(bcenters))
            bins=np.histogram(np.hstack((tapiti,bciti)), bins=30)[1] #get the bin edges
            sns.distplot(tapiti, hist=True, bins=bins, ax=ax[j,1], kde=True, color='red', label='taps')
            sns.distplot(bciti, hist=True, bins=bins, ax=ax[j,1], kde=True, color='blue', label='stim')
        
                        
            ##### GAT Pulse Analysis ####
            frmsds_t, freqbins_t = frmsd_iso(taps, shift_res=60, freq_res=30)
            frmsds_bc, freqbins_bc = frmsd_iso(s2t(bcenters), shift_res=60, freq_res=30)                        
            ## PLOT GAT
            plotGAT(frmsds_t, frmsds_bc, freqbins_t, ax[j,2])
            
            #### DO AUTOCORRELATION FOR TAPS and BCENTERS #####
            bc_60= distributeTrigs(s2t(bcenters)) # resample to 60 Hz, place narrow npdf conv trigs at idxs
            taps_60 = distributeTrigs(taps)       # resample to 60 Hz, etc.             
            # perform autocorrelation            
            ac_bc = ccorr(bc_60, bc_60)
            ac_taps = ccorr(taps_60, taps_60)           
            ax[j,3].plot(ac_bc, color='blue', linewidth=0.7, label='ac stim')
            ax[j,3].plot(ac_taps, color='red', linewidth=0.7, label='ac taps')
            #ax[j,3].legend() 
 
            # perform cross correlation            
            cc = ccorr(bc_60, taps_60)
            ax[j,4].plot(cc, linewidth=0.7, label='cross correlation')
            
        ax[j, 1].legend()
        ax[j, 2].legend()
        ax[j, 3].legend()
        ax[j, 4].legend()
        ax[j, 1].set_xlabel('ITI (secs)')
        ax[0, 1].set_title('ITI histogram')
        ax[0, 2].set_title('GAT Pulse Analysis')
        ax[0, 3].set_title('autocorrelation of taps vs. beat centers')
        
        couplingcond = sc[0].split('_')[0]
        fig.suptitle(person + ' ' + couplingcond)
        fig.tight_layout()
        # fig.savefig(os.path.join(plotdir, person, f'{couplingcond} data stats.png'), dpi=150)
        #fig.savefig(f'/Users/nolanlem/Desktop/gat-plots/{person} ' + couplingcond + '.png', dpi=150)
end_time = time.time()
print(f'time for one subject: %r sec' %(end_time-start_time) )           
#%%  same as before just selected people and plots 
### generate single examples of 

plotdir = '/Users/nolanlem/Documents/kura/swarmgen/mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/'

start_time = time.time()

stim = 'weak_n_72_1'
stim = 'strong_n_72_2'
stim = 'medium_n_81_2'

stim = 'medium_n_72_1'

thesubject = imitators[0]
thesubject = imitators[1]


thesubject = densesubs[0]
stim = 'none_n_72_1'

thesubject = densesubs[-1]
stim = 'none_n_72_1'
stim = 'none_n_72_3'


for n,person in enumerate([thesubject]):

    print(f'working on {person} %r/%r'%(n, len(subject)))   
    v = person.split('_')[1].split('-')[1][-1]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5), sharey='col', sharex='col')
        
    ax[2].invert_yaxis() # invert y axis for GAT pulse analysis 

    print(f'\t working on {person} in {stim}')

    stim_txt = reformatStim(stim)
    
    bbfile = 'stim-no-timbre-5/stimuli_' + stim.split('_')[-1] + '/phases/beat-windows/' + stim_txt
    
    bbins = np.loadtxt(bbfile) # already in samples
    bbinsiti = np.diff(bbins)/2
    bcenters = bbins[:-1] + bbinsiti #make beat centers

    y, sr_ = librosa.load('./allstims/' + stim + '.wav')
        
    taps = subject_resps[person][stim]
    # if tap array is empty for some reason fill it with np.nan
    if len(taps) == 0:
        taps = np.array([0])
    taps_s = t2s(taps) # taps in secs to samples
    ##### PLOT STI ####
    plotStim(y, taps_s, bbins, bcenters, ax[0], stim)
    #####  PLOT HISTO for TAPS and STIM OVERLAYED####
    tapiti = np.diff(taps)
    bciti = np.diff(s2t(bcenters))
    bins=np.histogram(np.hstack((tapiti,bciti)), bins=30)[1] #get the bin edges
    sns.distplot(tapiti, hist=True, bins=bins, ax=ax[1], kde=True, color='red', label='taps')
    sns.distplot(bciti, hist=True, bins=bins, ax=ax[1], kde=True, color='blue', label='stimulus beat center')

                
    ##### GAT Pulse Analysis ####
    frmsds_t, freqbins_t = frmsd_iso(taps, shift_res=60, freq_res=30)
    frmsds_bc, freqbins_bc = frmsd_iso(s2t(bcenters), shift_res=60, freq_res=30)                        
    ## PLOT GAT
    plotGAT(frmsds_t, frmsds_bc, freqbins_t, ax[2])
            

    ax[1].legend()
    ax[2].legend()
    ax[0].set_xlabel('seconds')
    
    ax[1].set_xlabel('ITI (secs)')
    ax[1].set_title('ITI histogram')
    ax[2].set_title('GAT Pulse Analysis')
    ax[2].set_xlabel('frequency (Hz)')
    ax[1].set_xlim([0,2.0])
    
    couplingcond = sc[0].split('_')[0]
    fig.suptitle(person + ' ' + couplingcond)
    fig.tight_layout()

end_time = time.time()
print(f'time for one subject: %r sec' %(end_time-start_time) )  

fig.savefig(os.path.join(plotdir, 'imitators-stats', f'{thesubject[-7:-4]} {stim} column data stats.png'), dpi=150)
#%%
fig,ax = plt.subplots(nrows=1,ncols=2)
sns.distplot(itis, hist=True, color='blue', ax=ax[0])
# xrange = np.arange(0, len(itis))
# xlabs = [np.round(elem/10.,2) for elem in xrange]
# ax[0].set_xticks(xrange)
# ax[0].set_xticklabels(xlabs)
#%% 
person='PARTICIPANT_kura-A2_2020-09-16_12h19.57.327.csv'
     
stim = 'medium_n_72_3'
stim_txt = reformatStim(stim)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,3))

# get beat windows
bbfile = 'stim-no-timbre-5/stimuli_' + stim.split('_')[-1] + '/phases/beat-windows/' + stim_txt
bbins = np.loadtxt(bbfile) # already in samples
bbinsiti = np.diff(bbins)/2
bcenters = bbins[:-1] + bbinsiti #make beat centers
#bcenters = s2t(bcenters)

# load stim wf
y, sr_ = librosa.load('./allstims/' + stim + '.wav')
# load subject taps
taps = subject_resps[person][stim]
taps_s = t2s(taps) # taps in secs to samples

ax[0].plot(y, linewidth=0.5) # plot stim wf
ax[0].vlines(taps_s, -1, -1.7, color='red', linewidth=0.7) # plot taps
ax[0].vlines(bbins, 0.5, -1.5, color='green', linewidth=0.7) # plot beat bins
ax[0].vlines(bcenters, -1.5 , 1, color='orange', linewidth=0.7) # plot beat centers
wfrange = np.arange(0, len(y))
xrange = np.arange(0, len(y), 22050)
ax[0].set_xticks(xrange)
xticklabs = [np.round(wfrange[elem]/22050.,2) for elem in xrange]
ax[0].set_xticklabels(xticklabs, fontsize=8)

#### HISTOGRAM OF ITIS ### ###############
tapiti = np.diff(taps)
sns.distplot(tapiti, hist=True, bins=20, ax=ax[1], axlabel='ITI (sec)', label='iti histogram')
ax[1].set_title('iti histogram')

############# GAT PULSE ANALYSIS ###########
frmsds_t, freqbins_t = frmsd_iso(taps, shift_res=60, freq_res=30)
frmsds_bc, freqbins_bc = frmsd_iso(s2t(bcenters), shift_res=60, freq_res=30)

i_frmsds_t = np.array(frmsds_t)*-1 # have to invert function to "find max" since we're plotting y ax inverted
i_frmsds_bc = np.array(frmsds_bc)*-1 
pkidx_t = findPk(i_frmsds_t, numpks=2)
pkidx_bc = findPk(i_frmsds_bc, numpks=4)

ax[2].invert_yaxis()
xrange = np.arange(0, len(freqbins_t),20)
ax[2].set_xticks(xrange)
# Hz? 
#xticklabs = [int(Hz2bpm(freqbins_t[elem])) for elem in xrange]
xticklabs = [np.round(freqbins_t[elem],2) for elem in xrange]

ax[2].set_xticklabels(xticklabs, fontsize=10)
ax[2].plot(frmsds_t, color='red', label='subject', linewidth=0.8)
ax[2].plot(frmsds_bc, color='blue', label='stimulus', linewidth=0.8)

# plot found peaks for both frmsds
for pk in pkidx_t:
    ax[2].plot(pk, frmsds_t[pk], marker='x', color='darkred')
    ax[2].vlines(pk, frmsds_t[pk], max(frmsds_t), color='darkred', linewidth=0.8, linestyle='-')
    pk_hz = np.round(freqbins_t[pk],2)
    ax[2].annotate(str(pk_hz),(pk, frmsds_t[pk]), color='darkred', fontsize=10)

for pk in pkidx_bc:
    ax[2].plot(pk, frmsds_bc[pk], marker='x', color='darkblue')
    ax[2].vlines(pk, frmsds_bc[pk], max(frmsds_bc), color='darkblue', linewidth=0.8, linestyle='-')
    pk_hz = np.round(freqbins_bc[pk],2)
    ax[2].annotate(str(pk_hz),(pk, frmsds_bc[pk]), color='darkblue', fontsize=10)

ax[2].set_xlabel('Hz')
ax[2].set_ylabel('frmsd')
ax[2].set_title('Pulse GAT Analysis')
ax[2].legend()
#### 



#ax[1].set_xlabel('bpm')


bc_60= distributeTrigs(s2t(bcenters))
taps_60 = distributeTrigs(taps)

ac_bc = ccorr(bc_60, bc_60)
ac_taps = ccorr(taps_60, taps_60)

ax[3].plot(ac_bc, color='blue', label='ac bc')
ax[3].plot(ac_taps, color='red', label='ac taps')
ax[3].legend()
if ac_bc.size > 0:
    ac_size = ac_bc.size
xrange = np.arange(0, len(ac_size))

fig.suptitle(f'{person} : {stim}')
fig.tight_layout()


#xticklabs = [str(np.round(freqbins[elem],2)) for elem in xlabs]
#ax.set_xticklabels(xticklabs, fontsize=8)
fig.savefig('/Users/nolanlem/Desktop/gat-plots/' + person + '-' + stim + '.png', dpi=150)
#%%


#%%
taps = np.arange(1,10)
#amt_noise = 0.1
#taps = taps + (-amt_noise + amt_noise*np.random.random(size=len(taps)))

bbins = np.arange(1, 11)
xrange = np.arange(1, 11)

f = interpolate.interp1d(xrange, bbins)

avgbbspacing = np.mean(np.diff(bbins))
shift_res = 10
shifts = np.linspace(0, 1, shift_res)

freqlow = 1 
freqhigh = 4 
freq_res = 10*(freqhigh-freqlow) + 1 # freq resolution between Hz 
freqshifts = np.linspace(freqlow, freqhigh, freq_res)



frmsd_hz = {}

for i,freq in enumerate(freqshifts):
    print(f'working on freq {freq} %r/%r'%(i, len(freqshifts)))
    bbins = np.arange(1, 11, 1/freq)
    frmsd = []
    for j, shiftamt in enumerate(shifts):
        print(f'\t working on shift: {shiftamt} %r/%r' %(j, len(shifts)))
        bbinshift = bbins + shiftamt
        offset = []
        for tap in taps:
            closest = find_nearest_diff(bbinshift, tap)
            offset.append(closest)
        print(offset)
        # get mean deviation 
        offsetrms = np.power(offset, 2)/freq
        offsetmx = np.nanmean(np.array(offsetrms))
        offsetmx = np.sqrt(offsetmx/len(offset)) 
        #print(taps, offset)
        offsetmx = np.nanmean(np.array(offset))
        #print(f'mx offset: {offsetmx}')
        frmsd.append(offsetmx)

    #print(f'frmsd at freq {freq} is {frmsd}')
    min_frmsd = min(frmsd)
    print(f'minimum frmsd is: {min_frmsd} at freq {freq}')
    frmsd_hz[str(freq)] = min_frmsd

frmsd_hz_ = []
for freq in freqshifts:
    frmsd_hz_.append(frmsd_hz[str(freq)])

#plt.plot(frmsd_hz.values())
xrange = np.arange(0, len(freqshifts),10)
ax.set_xticks(xrange)
xticklabs = [np.round(freqshifts[elem],2) for elem in xrange]
ax.set_xticklabels(xticklabs)
ax.plot(frmsd_hz_)
ax.set_xlabel('Hz')
ax.set_ylabel('frmsd')
ax.set_title('Pulse GAT Analysis')


#%% plots: 
rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all'
csvfile = os.path.join(rootdir, 'all-stim-taps-stats-r-cp-iti.csv')
df = pd.read_csv(csvfile) 
#%%
avgtaps = df['avg num taps per bin'].values
Rmag = df['R mag'].values
centerbpms = df['center bpm'].values
itis = df['norm iti'].values
itis_10 = [elem if elem < 10 else np.nan for elem in itis]
stds = df['norm std'].values
stds_lim = [std if std < 2 else np.nan for std in stds]
npvis = df['nPVI'].values


# 2D plots 
fig, ax = plt.subplots(nrows=9, ncols=1, figsize=(5,20))
# |R| vs. avg num taps per bin 
ax[0].plot(Rmag, avgtaps, linestyle='None', marker='.', linewidth=0.5, alpha=0.5)
ax[0].set_ylabel('avg number of taps per beat bin')
ax[0].set_xlabel('phase coherence magnitude |R| ')

# center bpms vs. avg num taps 
ax[1].plot(centerbpms, avgtaps, linestyle='None', marker='.', linewidth=0.5, alpha=0.5)
ax[1].set_xlabel('center bpms')
ax[1].set_ylabel('average number of taps per beat bin')

# |R| vs. norm avg iti
ax[2].plot(Rmag, itis_10, linestyle='None', marker='.', linewidth=0.5, alpha=0.5)
ax[2].set_xlabel('|R|')
ax[2].set_ylabel('avg ITI')

# |R| vs. norm avg std of iti
ax[3].plot(Rmag, stds_lim, linestyle='None', marker='.', linewidth=0.5, alpha=0.5)
ax[3].set_xlabel('|R|')
ax[3].set_ylabel('STD ITI')

ax[4].plot(stds_lim, avgtaps, linestyle='None', marker='.', linewidth=0.5, alpha=0.3)
ax[4].set_xlabel('STD ITI')
ax[4].set_ylabel('average num taps per bin')

ax[5].plot(itis_10, npvis, linestyle='None', marker='.', linewidth=0.5, alpha=0.3)
ax[5].set_xlabel('avg ITI')
ax[5].set_ylabel('nPVI')

ax[6].plot(stds_lim, npvis, linestyle='None', marker='.', linewidth=0.5, alpha=0.3)
ax[6].set_xlabel('STD ITI')
ax[6].set_ylabel('nPVI')

ax[7].plot(npvis, avgtaps, linestyle='None', marker='.', linewidth=0.5, alpha=0.3)
ax[7].set_xlabel('nPVI')
ax[7].set_ylabel('avg num taps per beat bin')

ax[8].plot(Rmag, npvis, linestyle='None', marker='.', linewidth=0.5, alpha=0.3)
ax[8].set_xlabel('Rmag')
ax[8].set_ylabel('npvis')

fig.tight_layout()

fig.savefig(os.path.join(rootdir, '2D distributions.png'), dpi=150)
#%% histogram of nPVI per coupling condition 
stims = df['stim'].values
cconds = ['none', 'weak', 'medium', 'strong']
npvi_cond = {'none': [], 'weak': [], 'medium': [], 'strong': []}
for stim, npvi_ in zip(stims, npvis):
    for cond in cconds:
        if stim.startswith(cond):
            npvi_cond[cond].append(npvi_)
    
#%% plot nPVIs per coupling condition 
fig, ax = plt.subplots(nrows=len(cconds), ncols=1, sharex=True, sharey=True)

for i, cond in enumerate(cconds):
    ax[i].hist(npvi_cond[cond], bins=100)
    ax[i].set_title(cond)
    
#%% 
from scipy import signal
from scipy.signal import find_peaks

def t2s(t):
    return librosa.time_to_samples(t)

def s2t(t):
    return librosa.samples_to_time(t)

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))



#subjects = list(set(df['subject'].values))
#thestim = 'strong_n_92_3'
thestim = 'medium_n_72_3'
#thestim = 'weak_n_92_3'
thestim_ = thestim.split('_')

person = 'PARTICIPANT_kura-A2_2020-09-16_12h19.57.327.csv'
taps = subject_resps[person][thestim]
bwfile = 'stim-no-timbre-5/stimuli_3/phases/beat-windows/' + thestim_[0] + '_' + thestim_[-2] + '_' + thestim_[-1] + '.txt'
bws = np.loadtxt(bwfile) # already in samples
bws = s2t(bws)
bcs = bws[:-1] + np.diff(bws)/2
bcs = np.array(bcs, dtype=np.int)
## 'none_n_92_3, subjects[33]

# resample at 60 Hz and distribute narrow npdf conv pulses 
taps_60 = distributeTrigs(taps)
bc_60 = distributeTrigs(bc)
#tapsidx = t2s(taps)
#tapsidx = [int(tap*sr_) for tap in taps]

y, sr_ = librosa.load(os.path.join('./allstims/', thestim + '.wav'))


plt.figure(figsize=(10,5))
#plt.plot(y, zorder=0)
plt.plot(bc_60, color='green')
plt.plot(taps_60, color='red', linewidth=1)

taps_cc = ccorr(taps_60, taps_60)
bc_cc = ccorr(bc_60, bc_60)
plt.plot(taps_cc, color='red', label='taps ac')
plt.plot(bc_cc, color='green', label='stim ac')
plt.legend()

#%% ###### perform autocorrelation  and cross correlation 
win = signal.windows.hann(3000) # 3000

filt_taps = signal.convolve(taps_s, win, mode='same') / sum(win)
filt_zcs = signal.convolve(bcenters, win, mode='same') / sum(win)
#plt.plot(filt_taps)

#### auto-correlation 
ac = np.correlate(filt_taps, taps_s, mode='full')
ac = ac[ac.size//2:]

#### cross-correlation with beat centers signal
# how much of tap ITI is in stim ITI beat   
cc = np.correlate(filt_taps, filt_zcs, mode='full')
cc = cc[cc.size//2:]

#plt.plot(ac[1:], linewidth=0.5)
plt.plot(cc[:], linewidth=0.5)

def findPk(sig, numpks=1):
    pk_list = []
        #height = max(sig[:])
    height = 0.001
    thepk = find_peaks(sig, height=height)
    #print(f'found peak at {thepk}') 

    return thepk

pks = findPk(cc, numpks=4)

plt.plot(cc, linewidth=1)
plt.vlines(pks[0], min(cc), max(cc), color='red', linewidth=0.6, linestyle='--')
#%%

from scipy import interpolate
bbins = np.arange(1, 11)
bbins = bbins + np.random.random(size=len(bbins))
xrange = np.arange(1, 11)
f = interpolate.interp1d(xrange, bbins)

freq = 2
period = 1/freq
xnew = np.arange(1, 10, period)
ynew = f(xnew)

plt.plot(xrange, bbins, '.', xnew, ynew, 'x')
#%% convert taps and beat bins to new sampling rate (60 Hz)
# taps = np.array(taps)
# taps_60 = taps*60
# taps_s = [np.int(elem) for elem in taps_60]
# taptrigs = np.zeros(max(taps_s)+1)
# for trig in taps_s:
#     taptrigs[trig] = 1
# plt.plot(taptrigs)
#%%
bwfile = 'stim-no-timbre-5/stimuli_3/phases/beat-windows/' + thestim_[0] + '_' + thestim_[-2] + '_' + thestim_[-1] + '.txt'
bws = np.loadtxt(bwfile) # already in samples
bws = s2t(bws)
bc = bws[:-1] + np.diff(bws)/2

bwfile = 'stim-no-timbre-5/stimuli_3/phases/beat-windows/' + thestim_[0] + '_' + thestim_[-2] + '_' + thestim_[-1] + '.txt'
bws = np.loadtxt(bwfile) # already in samples
bws = s2t(bws)
bc = bws[:-1] + np.diff(bws)/2
bc_60 = bc*60
bc_s = [np.int(elem) for elem in bc_60]
bctrigs = np.zeros(max(bc_s)+1)
for trig in bc_s:
    bctrigs[trig] = 1
plt.plot(bctrigs*2-1)

 ###### perform autocorrelation  and cross correlation 
win = signal.windows.hann(10) # 3000

filt_taps = signal.convolve(taptrigs, win, mode='same') / sum(win)
filt_zcs = signal.convolve(bctrigs, win, mode='same') / sum(win)
cc = np.correlate(filt_taps, filt_zcs, mode='full')
cc = cc[cc.size//2:]

ac_t = np.correlate(filt_taps, filt_taps, mode='full')
ac_t = ac_t[ac_t.size//2:]

ac_zcs = np.correlate(filt_zcs, filt_zcs, mode='full')
ac_zcs = ac_zcs[ac_zcs.size//2:]

#plt.plot(ac[1:], linewidth=0.5)
plt.plot(cc[:], linewidth=0.5)
plt.plot(ac_t, linewidth=0.5, color='red', label='ac(taps)')
plt.plot(ac_zcs, linewidth=0.5, color='purple', label='ac(stim)')
plt.legend()

#%%

        


#%%

tapinterpfunc = interp1d(np.arange(0, len(taps)), taps)

numbeats = 19
p_sr = 60. 
newxrange = np.arange(0, numbeats*p_sr)
ynew = tapinterpfunc(newxrange)



#%% 3D plot
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')


ax.scatter3D(avgtaps, Rmag, centerbpms, c=centerbpms, cmap='Greens', alpha=0.5, marker='.')

ax.set_xlabel('avg number taps per beat bin')
ax.set_ylabel('|R|')
ax.set_zlabel('bpm');
plt.savefig(os.path.join(rootdir, '3D distributions.png'), dpi=150)
#%%

def getKMeans(x, y nclusters=2):

    kmeans = KMeans(n_clusters= nclusters)
    
    points = np.array([x, y]).T
    lendata = float(points.shape[0])  
    # fit kmeans object to data
    means.fit(points)
    y_km = kmeans.fit_predict(points)
    klen = []
    for k in range(numclusters):
        plt.scatter(points[y_km == k,0], points[y_km == k,1], s=10, alpha=0.5) 
        klen.append(len(points[y_km == k,0]))
        print(f'category {k} had %r datapoints which was %r percent of distribution' %(klen[k], klen[k]/lendata))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, f'K-mean-{numclusters}-clusters.png'), dpi=150)
#%%
#########################################
########### K-MEANS CLUSTERING  #########
### |R| vs. AVG TAPS
from sklearn.cluster import KMeans

# create kmeans object
numclusters = 2
kmeans = KMeans(n_clusters= numclusters)

points = np.array([Rmag, avgtaps]).T
lendata = float(points.shape[0])

# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(points)

klen = []
for k in range(numclusters):
    plt.scatter(points[y_km == k,0], points[y_km == k,1], s=10, alpha=0.5) 
    klen.append(len(points[y_km == k,0]))
    print(f'category {k} had %r datapoints which was %r percent of distribution' %(klen[k], klen[k]/lendata))


plt.xlabel('|R|')
plt.ylabel('avg number of taps per beat bin')
plt.tight_layout()
plt.savefig(os.path.join(rootdir, f'K-mean-R-taps_{numclusters}-clusters.png'), dpi=150)
#%% 
####### centerbpm vs. AVG TAPS
numclusters = 3
kmeans = KMeans(n_clusters= numclusters)

points = np.array([avgtaps, centerbpms]).T
# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(points)

klen = []
for k in range(numclusters):
    plt.scatter(points[y_km == k,0], points[y_km == k,1], s=10, alpha=0.5) 
    klen.append(len(points[y_km == k,0]))
    print(f'category {k} had %r datapoints which was %r percent of distribution' %(klen[k], klen[k]/lendata))


plt.xlabel('center bpm')
plt.ylabel('avg number of taps per beat bin')
plt.tight_layout()
plt.savefig(os.path.join(rootdir, 'K-mean-centerbpm-taps_{numclusters}-clusters.png'), dpi=150)   
#%%%
############# AGGLOMERATIVE CLUSTERING
### |R| vs. AVG TAPS 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

points = np.array([Rmag, avgtaps]).T

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)

#%%
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=10, alpha=0.5)
plt.xlabel('|R|')
plt.ylabel('avg number of taps per beat bin')
plt.tight_layout()
plt.savefig(os.path.join(rootdir, 'Agg-cluster-R-taps.png'), dpi=150) 
#%%
points = np.array([centerbpms, avgtaps]).T

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)
#%%
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=10, alpha=0.5)
plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=10, alpha=0.5)
plt.xlabel('|R|')
plt.ylabel('avg number of taps per beat bin')
plt.tight_layout()
plt.savefig(os.path.join(rootdir, 'Agg-cluster-centerbpm-taps.png'), dpi=150)  

#%% 





