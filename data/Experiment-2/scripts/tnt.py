#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:59:41 2022
NB:
path to stimuli: ../../swarmgen/mturk-csv/allstims/
@author: nolanlem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 
import csv
from sklearn.datasets import make_blobs
os.chdir('/Users/nolanlem/Documents/kura/swarmgen/')
import librosa 
from scipy import signal
from scipy.signal import find_peaks
from scipy import stats
import time 
from itertools import chain 
import glob
sns.set_theme('notebook', 'darkgrid')
from fun.functions import *

from astropy.stats import rayleightest
import matplotlib.cm as cm


os.chdir('/Users/nolanlem/Documents/kura/final-tapping-scripts/timbre-paper/')



#%% some functions 
def str2nparr(dfslice):
    result = dfslice.apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
    return result

def str_to_array(s):
    s = s.replace('[', '').replace(']', '').replace('\n', '').split()
    return np.array([float(x) for x in s])

def tapSelect(tapsarr, bbin, beatcenters, optimize='random'):
    for i, tap in enumerate(tapsarr):
        if tap is not np.nan and (tap.size >= 1):
            tap[tap>beatcenters[-1]] = np.nan

            if optimize=='random':
                tapsarr[i] = tap[np.random.randint(low=0, high=len(tap))]                
            else:
                tapsarr[i] = tap
            if optimize == 'nearest':
                nearest_tap = return_nearest_idx(tap, beatcenters[i])
                tapsarr[i] = nearest_tap
            if optimize=='none':
                tapsarr[i] = tap
        else:
            tapsarr[i] = np.nan      

    return tapsarr

#%%
def parseString(st):

    _ = st.replace('[','').replace(']','')
    _ = _.split(' ')
    _ = [x for x in _ if x]
    _ = [float(x) for x in _]
        
    return np.array(_)


#%%

# df_ = pd.read_csv('/Users/nolanlem/Documents/kura/swarmgen/mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/csvs/all-stats-4-7-nt-new.csv')

df = pd.read_csv('./df-bb-taps-3-2.csv')

df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]

#%% check mean raw iti for 3d and 3s tap trials 

# all
dfslice = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3d')]['taps']



subject_cat = 'all'

dfslice3d = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3d')]['taps']
dfslice3s = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3s')]['taps']


#dfslice3s = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3s') & (df['subject cat'] == subject_cat)]['taps']


# for 3d mx(sd)
dfd = str2nparr(dfslice3d)
for d in dfd:
    mean_raw_iti = np.mean(np.diff(d))
    if mean_raw_iti < 1:
        mxsd.append(mean_raw_iti)
densemx = np.mean(mxsd)
densesx = np.std(mxsd)

print(f'3d: {subject_cat}, 3d raw iti mx: {densemx} ({densesx})')

        
# for 3s mx(sd)
dfd = str2nparr(dfslice3s)

for d in dfd:
    mean_raw_iti = np.mean(np.diff(d))
    if mean_raw_iti < 1:
        mxss.append(mean_raw_iti)
        
densemx = np.mean(mxss)
densesx = np.std(mxss)
print(f'3s: {subject_cat}, 3s raw iti mx: {densemx} ({densesx})')

print('\n')
import scipy.stats as stats

stats.ttest_ind(a=mxsd, b=mxss, equal_var=False)

# by subject category 
subcats = ['regular', 'hybrid', 'fast']
for sc in subcats:
    df3d = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3d') & (df['subject cat'] == sc)]['taps']
    df3s = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3s') & (df['subject cat'] == sc)]['taps']
    
    mxsd, mxss = [],[]

    dfd3d = str2nparr(df3d)
    dfd3s = str2nparr(df3s)
    
    for d in dfd3d:
        mean_raw_iti = np.mean(np.diff(d))
        if mean_raw_iti < 1:
            mxsd.append(mean_raw_iti)
    densemx = np.mean(mxsd)
    densesx = np.std(mxsd)
    print(f'3d: {sc}, {ds} raw iti mx: {np.round(densemx,3)} ({np.round(densesx,3)})')


    for d in dfd3s:
        mean_raw_iti = np.mean(np.diff(d))
        if mean_raw_iti < 1:
            mxss.append(mean_raw_iti)
    densemx = np.mean(mxss)
    densesx = np.std(mxss)
    print(f'3d: {sc}, {ds} raw iti mx: {np.round(densemx,3)} ({np.round(densesx,3)})')



#%%
ds ='3d'
df[(df['timbre'] == 'n') & (df['dispersion group'] == ds)]['taps']

#%%
df = pd.read_csv('./df-bb-taps-3-2.csv')
# Define a function to convert string representations of numpy arrays to actual numpy arrays


df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]

def str_to_nparray(str_array):
    # Remove the square brackets and split the string on whitespace
    str_array = str_array.strip('[]')
    str_array = str_array.split()

    # Convert the list of strings to a numpy array of floats
    np_array = np.array(str_array, dtype=float)

    return np_array

df['taps'] = df['taps'].apply(str_to_nparray)

#%%


#%%


# os.chdir('/Users/nolanlem/Documents/kura/final-tapping-scripts/timbre-paper/')
# tappingdata = 'df-bbtaps-11-1.csv'
# df = pd.read_csv(tappingdata)

cps = ['strong', 'medium', 'weak', 'none']
#%% get mean raw iti for 3d and 3s
### NB don't need to run 

dfslice = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3d') & (df['mean raw iti'] < 0.25)]['taps']

mxsd, mxss = [],[]
dfd = str2nparr(dfslice)
for d in dfd:
    mxsd.append(np.mean(np.diff(d)))

print(f'dense raw iti mx: {np.mean(mxsd)}')

print(dfslice.shape)

dfslice = df[(df['timbre'] == 'n') & (df['dispersion group'] == '3s') & (df['mean raw iti'] < 0.25)]['taps']

print(dfslice.shape)

dfs = str2nparr(dfslice)
for d in dfs:
    mxss.append(np.mean(np.diff(d)))

print(f'sparse raw iti mx: {np.mean(mxss)}')

#%%

# df['mean raw iti'] = ''

# for row, col in df.iterrows():
#     tps = col['taps']
#     tps = parseString(tps)
#     mxtps = np.mean(np.diff(tps))
#     print(mxtps)
    
#     df.at[row,'mean raw iti'] = mxtps
# #%%

# df.to_csv('./df-bb-taps-3-2.csv')    

#%% create subject groups
reg_reg = list(set(df[df.subcat == 'regreg']['subject']))
hyb_reg = list(set(df[df.subcat == 'hybreg']['subject']))
hyb_hyb = list(set(df[df.subcat == 'hybhyb']['subject']))
fast_fast = list(set(df[df.subcat == 'fastfast']['subject']))


#%% load stimuli zero crossings from .npy files 

tap_optimizer = 'none' # 'random' - choose random tap, 'nearest' - choose optimal tap 
numbeats = 19

stimdirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']

stimtimbredir = '../../swarmgen/stim-no-timbre-5/'
unique_stimuli = sorted(list(set(df['stim'])))

nbatch = set(list(df[(df.coupling == 'none') & (df.timbre == 'n')]['stim']))
wbatch = set(list(df[(df.coupling == 'weak') & (df.timbre == 'n')]['stim']))
mbatch = set(list(df[(df.coupling == 'medium') & (df.timbre == 'n')]['stim']))
sbatch = set(list(df[(df.coupling == 'strong') & (df.timbre == 'n')]['stim']))


# nbatch = [elem for elem in unique_stimuli if elem.startswith('n')]
# wbatch = [elem for elem in unique_stimuli if elem.startswith('w')]
# mbatch = [elem for elem in unique_stimuli if elem.startswith('m')]
# sbatch = [elem for elem in unique_stimuli if elem.startswith('s')]
cpbatches = [nbatch, wbatch, mbatch, sbatch]

# create dictionary to hold zero crossings for timbre conds and coupling conds
zcs = {'nt' : {'strong': [], 'medium': [], 'weak' :[], 'none' :[]}, 't': {'strong': [], 'medium': [], 'weak': [], 'none': [] }}

#for tnt, tstr in zip(['nt', 't'],['no-timbre', 'timbre']):
for tnt, tstr in zip(['nt'],['no-timbre']):

    stimtimbredir = '../../swarmgen/' + 'stim-' + tstr + '-5'

    for cpbatch, cpstr in zip(cpbatches, ['none', 'weak', 'medium', 'strong']):
        zcs[cpstr] = []
        for i, stim in enumerate(cpbatch):
            print(f'working on %r/%r'%(i+1, len(cpbatch)))    
            bname = reformatStim(stim)
            print(bname)    
            trigsdir = os.path.join(stimtimbredir, 'stimuli_' + bname[-1], 'trigs', bname + '.npy')
            bbin = str2nparr(df[df['stim'] == stim]['beat windows']).iloc[0]
            beatcenters = bbin[:-1] + np.diff(bbin)/2
            zcs_ = getTrigs(trigsdir)
            
            circlerange = [elem*2*np.pi for elem in range(len(bbin))]
            circlemap = interp1d(bbin, circlerange)
        
            zcsbin = []
            for zc in zcs_:
                zc = np.delete(zc, np.where(zc>bbin[-1])) # remove zcs below first bb
                zc = np.delete(zc, np.where(zc<bbin[0]))  # remove zcs above last bb 
                binnedzcs = binBeats(zc, bbin)
                binnedzcs = binnedzcs[:numbeats]
                cbinnedzcs = tapSelect(binnedzcs, bbin, beatcenters, optimize=tap_optimizer)
                cbinnedzcs = np.hstack(cbinnedzcs)
                
                zcsbin.append(list(circlemap(cbinnedzcs)%(2*np.pi)))
            zcs[tnt][cpstr].extend(zcsbin)
#%% 
# repeaters and upgraders list of subjects 
hyb_reg = list(set(df[(df['subcat'] == 'hybreg')]['subject']))
hyb_hyb = list(set(df[(df['subcat'] == 'hybhyb')]['subject']))
reg_reg = list(set(df[(df['subcat'] == 'regreg')]['subject']))
fast_fast = list(set(df[(df['subcat'] == 'fastfast')]['subject']))


#%%
########## make tap by tap csv file to compare tap distributions for only k=1+2 taps for repeaters-upgraders
##########################################################################################

fi = open('./csvs/binned-taps-k12-subcat.csv', 'w')
writer = csv.writer(fi)
writer.writerow(['tnt', 'subcat', 'coupling', 'btaps'])

dfk12 = df[((df['k group'] == 0) | (df['k group'] == 1))]
dfreg = dfk12[dfk12.subcat == 'regreg']
dfhreg = dfk12[dfk12.subcat == 'hybreg']
dfhyb = dfk12[dfk12.subcat == 'hybhyb']
dffast = dfk12[dfk12.subcat == 'fastfast']

for dfbatch, subcatstr in zip([dfreg, dfhreg, dfhyb, dffast], ['regreg', 'hybreg', 'hybhyb', 'fastfast']):
    
    for tnt in ['n', 't']:
        for cp in cps:
            taps = dfbatch[(dfbatch.timbre == tnt) & (dfbatch.coupling == cp)]['bbtaps']
            taps = str2nparr(taps)
            flattenedtaps = np.concatenate(taps.values).ravel()
            #print(flattenedtaps.shape)
            for tp in flattenedtaps:
                writer.writerow([tnt, subcatstr, cp, tp])

    
fi.close()

#%% 
########### make swarm plots ######################
##################################################################

ntreg = list(set(df[(df['subject cat'] == 'regular') & (df.timbre == 'n')]['subject']))
nthyb = list(set(df[(df['subject cat'] == 'hybrid') & (df.timbre == 'n')]['subject']))
ntfast = list(set(df[(df['subject cat'] == 'fast') & (df.timbre == 'n')]['subject']))



r_psi_tnt_fi = 'R-psi-tnt-k12.csv'

fi = open('./csvs/' + r_psi_tnt_fi,'w')
writer = csv.writer(fi)
writer.writerow(['taponset', 'subcat', 'tnt', 'coupling', 'r', 'psi'])

n_per_cat = getNumberSubjectsPerCat(df)

fig_agg, ax_agg = plt.subplots(nrows=3, ncols=4, subplot_kw=dict(polar=True), 
                            figsize=(10,10), 
                            sharex=True)
for ax_ in ax_agg.flat:
    ax_.set_thetagrids([])
    ax_.set_yticklabels([])
    ax_.set_axisbelow(True)
    ax_.grid(linewidth=0.1, alpha=1.0)

style = dict(size=10, color='gray')
  
# uncomment for doing non-changers 
k = 0

tntdf = 'n'
#for tnt, tntdf in zip(['nt', 't'], ['n', 't']):
# uncomment for doing all t or nt
for k, tnt in enumerate(['nt']):
    i = 0

    # for non-changers, and hyb->reg group 
    # for scat, scatstr in zip([reg_reg, hyb_reg, hyb_hyb, fast_fast], ['regular', 'hybreg', 'hybrid', 'fast']):
    
    # uncomment only timbre 
    for scat, scatstr in zip([ntreg, nthyb, ntfast], ['regular', 'hybrid', 'fast']):
        numsubjects = len(list(set(df[df['subject'].isin(scat)]['subject'])))
        
        for j, cp in enumerate(['strong', 'medium', 'weak', 'none']):
            
            ##### model swarm zcs ####           
            aggzcs = np.concatenate(zcs[tnt][cp], 0)
            aggzcs = aggzcs[~np.isnan(aggzcs)]
            rm, psim = getCircMeanVector(aggzcs)
            
            aggzcs -= psim
            noise = 0.3*np.random.random(size=len(aggzcs))
            c = cm.Reds(np.linspace(0, 1, len(aggzcs)))
            
            # plot stim swarm?
            ax_agg[i,j].scatter(aggzcs, 1 - noise, s=20, alpha=0.02, color=c, marker='.', edgecolors='none', linewidth=0.5)
            ax_agg[i,j].arrow(0, 0.0, 0, rm, color='firebrick', linewidth=1, zorder=2, label='stimulus onsets')         
            
            # only k=1+2 taps ?
            aggtaps = df[(df['subject'].isin(scat)) & (df.timbre == tntdf) & (df.coupling == cp)]['bbtaps']
            #aggtaps = np.concatenate(sctaps[tnt][cp][scatstr])
            #aggtaps = [elem for elem in aggtaps if np.isnan(elem) == False]
            #aggtaps = np.array(aggtaps)
            aggtaps = str2nparr(aggtaps)
            aggtaps = np.concatenate(aggtaps.values).ravel()
            aggtaps += psim # push by psi_stim to center them wrt zcs of stimuli
            
            #aggtaps_ = np.reshape(aggtaps%(2*np.pi), (1, len(aggtaps)))
            
            p_subj = rayleightest(aggtaps%(2*np.pi))
            p_stim = rayleightest(aggzcs%(2*np.pi))
            
            #['taponset', 'subcat', 'tnt', 'coupling', 'psi'])

            # uncomment if writing all tap onsets to csv file
            # for tp in aggtaps:
            #     writer.writerow(['tap', scatstr, tnt, cp, tp])
            # for tp in aggzcs:
            #     writer.writerow(['onset', scatstr, tnt, cp, tp])
            
            # uncomment if writing tap/onset R,psi to csv file 
            writer.writerow(['onset', scatstr, tnt, cp, rm, psim])

            
            # if p_subj > 0.05:
            #     print('subject taps NS directionality')
            print(f'{tnt} {cp} p_stim/p_subj: {p_stim}/{p_subj}' )
            
            rs, psis = getCircMeanVector(aggtaps)
            
            writer.writerow(['tap', scatstr, tnt, cp, rs, psis])

            #psis -= psim 
            noise = 0.3*np.random.random(size=len(aggtaps))
            
            
            #rose_plot(ax_agg[i,j], aggtaps, zorder=0)
            
            displace = 0.8
            color = 'darkblue'
            c = cm.Blues(np.linspace(0, 1, len(aggtaps)))
            tnt_str = 'no timbre'

            if k == 1:
                displace = 0.8-0.3
                color = 'darkred'
                c = cm.Reds(np.linspace(0,1,len(aggtaps)))
                tnt_str = 'timbre'
                
            ax_agg[i,j].scatter(aggtaps, displace - noise, s=20, alpha=0.05, color=c, marker='.', edgecolors='none', linewidth=0.5)
            ax_agg[i,j].arrow(0, 0.0, psis, rs, color=color, linewidth=1,  zorder=2, label=f'{tnt}') 
            # ax_agg[i,j].arrow(0, 0, 0, 1, color='black', linewidth=0.2, 
            #                   linestyle='--')
            ax_agg[0,j].set_title(f'{cp}') 
            
            #ax_agg[i,j].text(1, 1, f'Rm={np.round(rm,2)} $\psi_m$={np.round(psim,2)}\nRs={np.round(rs,2)} $\psi_s$={np.round(psis,2)}', fontsize=8)
            
        ax_agg[i,0].set_ylabel(f'{scatstr} ') 
        #ax_agg[i,3].legend()
        i+=1 
    k+=1
plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')

plt.tight_layout()
fig_agg.savefig('./saved-plots/swarm-plots-only-k12.png', dpi=150)
fig_agg.savefig('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/pc-plots-rev.png', dpi=150)
fig_agg.savefig('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/pc-plots-rev.eps', dpi=150)


fi.close()

#%%
#%%  
############ read from csv file made from aggregate plots make nt/t bar plots for |R| per repeater upgrader subject group ############
#########################################################################################
r_psi_tnt_fi = './csvs/R-psi-tnt-k12.csv' # just generated above 

data = pd.read_csv(r_psi_tnt_fi)

fig,ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(10,5))        

ru_subcats = ['regular', 'hybreg', 'hybrid', 'fast']

ntxcoords = np.array([0.1,0.2,0.3,0.4])
width = 0.025
txcoords = ntxcoords + width

for i, subcat in enumerate(ru_subcats):
    rmxnt, rmxt = [], []
    for cp in cps:
        ntslice = data[(data.coupling == cp) & (data.tnt == 'nt') & (data.subcat == subcat) & (data.taponset == 'tap')]['r']

        tslice = data[(data.coupling == cp) & (data.tnt == 't') & (data.subcat == subcat) & (data.taponset == 'tap')]['r']
     
        
        rmxnt.append(np.nanmean(ntslice.values))
        rmxt.append(np.nanmean(tslice.values))
        
            
        print(f'{subcat} {cp} R_nt = {rmxnt} R_t = {rmxt}')
    
    ax[i].bar(ntxcoords, rmxnt, color='darkblue', width=width, label='nt')
    ax[i].bar(txcoords, rmxt, color='darkred', width=width, label='t')
    ax[i].set_ylabel('|R|')
    ax[i].set_xticks(ntxcoords)
    ax[i].set_xticklabels(cps, rotation=45)
    ax[i].set_title(f'{subcat}')
    ax[i].set_ylim([0,1])

plt.suptitle('differences in mean |R| between nt, t taps per group')    
plt.legend()

plt.savefig('./saved-plots/|R| nt t subgroup.png', dpi=140)

#%% 


#%% time course for all subjects in t ,nt
### fill up arrays etc to do time course plots for all subjects
##### separated by subgroup (repeaters upgraders )

treg = list(set(df[(df['subcat'] == 'regreg')]['subject']))
nthyb_treg = list(set(df[(df['subcat'] == 'hybreg')]['subject']))
thyb = list(set(df[(df['subcat'] == 'hybhyb')]['subject']))
tfast_rev = list(set(df[(df['subcat'] == 'fastfast')]['subject']))


fi = open('./csvs/tnt-niti-beat-segments-per-subject.csv', 'w')
writer = csv.writer(fi)
#writer.writerow(['subject', 'tnt', 'coupling', 'subcat', '1mx', '2mx', '3mx', '4mx', '5mx', '6mx', '7mx','1sx', '2sx', '3sx', '4sx', '5sx', '6sx', '7sx'])

writer.writerow(['subject', 'tnt', 'coupling', 'subcat', 'bs', 'mx', 'sd'])

# remove outlier participant from fast group ?
#tfast_rev = tfast
#tfast_rev.remove('PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv')

beatslices = [(0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,21)]
bslices = [['b' + str(x) for x in np.arange(btslice[0], btslice[1])] for btslice in beatslices]

# defined dictionaries of lists to hold mx,sx per bs per subject
nitimx, nitisd, nitimxerr, nitisderr = {},{},{},{}
for tnt in ['n', 't']:
    nitimx[tnt], nitisd[tnt], nitimxerr[tnt], nitisderr[tnt] = {},{},{},{} 
    for sb in ['regular', 'hybreg', 'hybrid', 'fast']:
        nitimx[tnt][sb], nitisd[tnt][sb], nitimxerr[tnt][sb], nitisderr[tnt][sb] = {}, {}, {},{}
        
        
# list of strings to seelct tap meaned data for each beat section in df
bslicesall = ['b' + str(elem) for elem in np.arange(22)]

for tnt in ['n', 't']:
    for subbatch, sbstr in zip([treg, nthyb_treg, thyb, tfast_rev], ['regular', 'hybreg', 'hybrid', 'fast']):
        for cp in cps:
            nitimx[tnt][sbstr][cp], nitimxerr[tnt][sbstr][cp] = [],[]
            nitisd[tnt][sbstr][cp], nitisderr[tnt][sbstr][cp] = [],[]
            #for subj in subbatch:
            tapslice = df[(df.subject.isin(subbatch)) & (df.timbre == tnt) & (df.coupling == cp)]
     
            stapmxs = [[] for _ in range(7)]
            stapsxs = [[] for _ in range(7)]
    
            for i,bts in enumerate(bslices):
                for subj in subbatch:
                    try:
                        subtapmx = np.nanmean(tapslice.loc[tapslice['subject'] == subj, bts])
                        subtapsx = np.nanstd(tapslice.loc[tapslice['subject'] == subj, bts])
                        
                    except:
                        subtapmx = np.nan
                        subtapsx = np.nan
                    
                    stapmxs[i].append(subtapmx)
                    stapsxs[i].append(subtapsx)
            
            stapmxs = np.array(stapmxs)
            
            for i,bts in enumerate(bslices):
                nitimx[tnt][sbstr][cp].append(np.nanmean(stapmxs[i]))
                nitisd[tnt][sbstr][cp].append(np.nanmean(stapsxs[i]))
                
                nitimxerr[tnt][sbstr][cp].append(np.nanstd(stapmxs[i]))
                nitisderr[tnt][sbstr][cp].append(np.nanstd(stapsxs[i]))
            
            # impute mx, sd
            imp_mx = np.nanmean(tapslice[bslicesall].values)
            imp_sx = np.nanstd(tapslice[bslicesall].values)
            
            # just for writing to csv for within subjects stats 
            for subj in subbatch:
                tapsarrmx, tapsarrsx = [], []
                for bs, bts in enumerate(bslices):
                    #try:
                    taps = tapslice.loc[tapslice['subject'] == subj, bts]
                    tapsmx = np.nanmean(taps)
                    tapssx = np.nanstd(taps)
                    if str(tapsmx) == 'nan': # if tapslice was completely empty fill w the impute mx,sx
                        #print(f'{subj} {cp} {bs} {tapsmx}, {tapssx}')
                        tapsmx = imp_mx
                        tapssx = imp_sx
                    writer.writerow([subj, tnt, cp, sbstr, bs, tapsmx, tapssx])
                    
                        
                    # except:
                    #     #print('here')
                    #     tapsmx = imp_mx
                    #     tapssx = imp_sx
                    #     writer.writerow([subj, tnt, cp, sbstr, bs, tapsmx, tapssx])

                        
                    tapsarrmx.append(tapsmx)
                    tapsarrsx.append(tapssx)
            
                    
                
fi.close()

#%% sanity check to make sure time course csv is correct
data = pd.read_csv('./csvs/tnt-niti-beat-segments-per-subject.csv')
bslicesall = ['b' + str(elem) for elem in np.arange(22)]


cols = ['blue', 'orange', 'green', 'red']
for tnt in ['n', 't']:
    fig,ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey='row', figsize=(10,6))

    i = 0
    for subbatch, subcatstr in zip([reg_reg, hyb_reg, hyb_hyb, fast_fast], ['regreg', 'hybreg', 'hybhyb', 'fastfast']):
        
        d = data[(data.subject.isin(subbatch)) & (data.tnt == tnt)]
        
        for n, cp in enumerate(cps):
            print(cp)
            mxs, sds, mxserr, sdserr = [], [], [], []
            for bt in np.arange(7):
                means = d[(d.bs == bt) & (d.coupling == cp)]['mx'].values
                stds = d[(d.bs == bt) & (d.coupling == cp)]['sd'].values

                mxslice = np.nanmean(means)
                mxerror = np.nanstd(means)
                
                
                sdslice = np.nanmean(stds)
                sderror = np.nanstd(stds)
                
                mxs.append(mxslice)
                sds.append(sdslice)
                
                if (sdslice - sderror) < 0:
                    bottomerr = sdslice 
                else:
                    bottomerr = sderror
                sderror_ = [bottomerr, sderror]
                
                mxserr.append(mxerror)
                sdserr.append(sderror_)
            
            xr = np.arange(0+n/7, 7+n/7, 1)

            ax[0,i].errorbar(x=xr, y=mxs, yerr=mxserr, capsize=2, marker='.', color=cols[n], linewidth=0.8, label=cp) 
            #ax[0,i].plot(mxs, label=cp, color=cols[n])
            ax[0,i].set_title(f'{tnt} {subcatstr}')
            ax[0,i].set_ylim([0,1.5])

            ax[1,i].errorbar(x=xr, y=sds, yerr=np.array(sdserr).T, capsize=2, marker='.', color=cols[n], linewidth=0.8, label=cp)             
            #ax[1,i].plot(sds, label=cp, color=cols[n])
            ax[1,i].set_title(f'{tnt} {subcatstr}')
            ax[1,i].set_ylim([-0.1,1.5])

        i += 1
    [ax_.set_ylabel('mean nITI') for ax_ in ax[:,0]]    
    for ax_ in ax[0,:]:
        ax_.set_xticks(np.arange(7))
        ax_.set_xticklabels([str(x+1) for x in np.arange(7)])
        ax_.set_xlabel('tap section')

   
#%% 
################################################################################                
#############################CURVE FITTING ###############################                
################################################################################



###  CURVE FITTING TO DECAYING EXPONENTIAL 
def model_func(t, A, K, C):
    return A * np.exp(K * t) + C

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C

# compute goodness of fit 
#### "coefficient of determination" (aka the R2 value)
def compute_gof(y, y_fit):
    ss_res = np.sum((y-y_fit)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


import scipy as sp
import scipy.optimize
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a*np.exp(b*x) + c

sns.color_palette("tab10")

# initialize dict of sx curves
axcurves, curveparams = {}, {}
for ts in ['n', 't']:
    axcurves[ts], curveparams[ts] = {}, {}
    for subcat in ['regular', 'hybreg', 'hybrid', 'fast']:
        axcurves[ts][subcat], curveparams[ts][subcat] = {}, {}
        for cp in cps:
            axcurves[ts][subcat][cp], curveparams[ts][subcat][cp] = 0, 0
            

t = np.arange(1,8)  # dep var, x-axis tap section vector  
fi = open(os.path.join('./csvs/', ts + '-params-curve-fit.csv'), 'w')
writer = csv.writer(fi)
writer.writerow(['timbre', 'group', 'coupling', 'A', 'B', 'R2'])

fig, ax = plt.subplots(nrows = 2, ncols = 4, sharex=True, sharey=True, figsize=(20,8))

    
for k, ts in enumerate(['t', 'n']): ### NB: only for nt, can change to t to do the same 

    
    for s, subcat in enumerate(['regular', 'hybreg', 'hybrid', 'fast']):
        for n, cp in enumerate(cps):
            print(f'working on {ts} {subcat} {cp}')
            C0 = nitisd[ts][subcat][cp][0]
            curve = nitisd[ts][subcat][cp]


            z = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  t,  curve)  
            print(f'\t {np.round(z[0][0],2)}*np.exp({np.round(z[0][1],3)}*t)')
            app_curve = z[0][0]*np.exp(z[0][1]*t)
            real_curve = np.copy(nitisd[ts][subcat][cp])
            r2 = compute_gof(real_curve, app_curve)
            print(f'{ts} {subcat} {cp} : {r2}')
            
            ax[k,s].plot(t, app_curve, label=cp, marker='x', color=sns.color_palette('tab10')[n]) 
            curveparams[ts][subcat][cp] = z[0]
            
            
            ## method 3
            # popt, pcov = curve_fit(func, t, curve, maxfev=5000)
            # ax[s].plot(t, func(t,popt[0], popt[1], popt[2]), label=cp, marker='x')
          
            ax[k,s].plot(t, curve, color=sns.color_palette('tab10')[n], linewidth=0.5)
            ax[k,s].set_xlabel('Tap Section')
            ax[k,s].set_title(subcat) 
            
            
            # write params to csv file 
            writer.writerow([ts, subcat, cp, z[0][0], z[0][1], r2])
        ax[k,s].set_title(f'{ts} {subcat}')



fig.suptitle(f'exp curve fitting on sx')

plt.savefig('./saved-plots/curve-fitting-nt-t.png', dpi=130)
            
fi.close()
#%% 
###################### create k cluster proportion table ########
########################################################################################

# find porportion trials in t in 3d and 3sf 
finame = './csvs/rep-upg-kcluster-percentages.csv'
fi = open(finame, 'w')
writer = csv.writer(fi)
writer.writerow(['tnt', 'subcat', 'coupling', '12mx', '12sd', '3mx', '3sx', '4mx', '4sx', '3dmx', '3dsd', '3smx', '3ssx'])

# by subject 
finame1 = './csvs/rep-upg-kcluster-percentages-subject.csv'
fi1 = open(finame1, 'w')
writer1 = csv.writer(fi1)
writer1.writerow(['subject', 'tnt', 'subcat', 'coupling', 'cnt12', 'cnt3', 'cnt3d', 'cnt3s', 'cnt4'])

# fi1 = open('./mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/csvs/t-kcluster-percentages.csv', 'w')
# writer = csv.writer(fi1)
# writer.writerow(['subject', 'subcat', 'coupling', '12mx', '12sd', '3mx', '3sx', '3dmx', '3dsd', '3smx', '3ssx'])

totals3d = df[(df.timbre=='t') & (df.coupling == cp) & (df['dispersion group'] == '3d')]

for subbatch, sbstr in zip([reg_reg, hyb_hyb, fast_fast, nthyb_treg], ['regular', 'hybrid', 'fast', 'hybreg']):
    totaltrialspercp = len(subbatch)*10
    for cp in cps:
        print(f'{sbstr} {cp}')
        cnttotal = len(subbatch)*40
        
        for tnt in ['n', 't']:
            cnt12p, cnt3p, cnt4p, cnt3dp, cnt3sp = [], [], [], [], []
            for subj in subbatch:
                # k1+2 cluster, 'beat extraction' distro
                cnt12 = df[(df.subject == subj) & ((df['k group'] == 0) | (df['k group'] == 1)) & (df.timbre == tnt) & (df.coupling == cp)].shape[0]
                # k=3 cluster, 'fast'
                cnt3 = df[(df.subject == subj) & (df['k group'] == 2) & (df.timbre == tnt) & (df.coupling == cp)].shape[0]
                
                # dense,sparse distros
                cnt3d = df[(df.subject == subj) & (df['dispersion group'] == '3d') & (df.timbre == tnt) & (df.coupling == cp)].shape[0]
                cnt3s = df[(df.subject == subj) & (df['dispersion group'] == '3s') & (df.timbre == tnt) & (df.coupling == cp)].shape[0]
                    
                # k=4,5 slow tapping
                cnt45 = df[(df.subject == subj) & ((df['k group'] == 3) | (df['k group'] == 4)) & (df.timbre == tnt) & (df.coupling == cp)].shape[0]
                                    
                #this code block for all t
                cnt12p.append(cnt12/10.)
                cnt3p.append(cnt3/10.)
                cnt3dp.append(cnt3d/10.)
                cnt3sp.append(cnt3s/10.)
                cnt4p.append(cnt45/10.)
                
                writer1.writerow([subj, tnt, sbstr, cp, cnt12/10., cnt3/10., cnt3d/10., cnt3s/10., cnt45/10.])
    
            # ca
            cnt12mx = np.round(np.mean(cnt12p), 4)
            cnt3mx = np.round(np.mean(cnt3p), 4)
            cnt12sx = np.round(np.std(cnt12p), 4)
            cnt3sx = np.round(np.std(cnt3p), 4)
            cnt4mx = np.round(np.mean(cnt4p), 4)
            cnt4sx = np.round(np.std(cnt4p), 4)
    
            
            cnt3dmx = np.round(np.mean(cnt3dp), 4)
            cnt3dsx = np.round(np.std(cnt3dp), 4)
            cnt3smx = np.round(np.mean(cnt3sp), 4)
            cnt3ssx = np.round(np.std(cnt3sp), 4)
            
            
            
            writer.writerow([tnt, sbstr, cp, cnt12mx, cnt12sx, cnt3mx, cnt3sx, cnt4mx, cnt4sx, cnt3dmx, cnt3dsx, cnt3smx, cnt3ssx])
        

        
        
        # count in 3d/3s
        # cnt3d = df[(df["dispersion group"]=='3d') & (df['subject cat'] == sbstr) & (df.coupling==cp) & (df.timbre=='t')].shape[0]
        # cnt3s = df[(df["dispersion group"]=='3s') & (df['subject cat'] == sbstr) & (df.coupling==cp) & (df.timbre=='t')].shape[0]
        # #cnttotal = df[(df['k group']== 2) & (df['subject cat'] == sbstr) & (df.timbre=='t')].shape[0]
        # cnttotal = len(subbatch)*40
        # if cnttotal != 0:
        #     print(f'\t 3d: {cnt3d} 3s: {cnt3s} %total: {np.round(100*(cnt3d+ cnt3s)/cnttotal,2)} %')
        # else:
        #     print(f'\t 3d: {cnt3d} 3s: {cnt3s} %total: NA')
            
        #writer.writerow([sbstr, cp, cnt12p, cnt3p])
            
fi.close()
fi1.close()

#%%
#%% 
########## plot kcluster proportions bar over bar vertically stacked 
finame = './csvs/rep-upg-kcluster-percentages.csv'

csvdf = pd.read_csv(finame) # finame from codeblock above

subcats = ['regular', 'hybreg', 'hybrid', 'fast']

y_offset = np.zeros(2)
index = np.arange(2) + 0.3
bar_width = 0.4
colors = plt.cm.BuPu(np.linspace(0, 0.5, 4))

# for subcat in subcats:
#     for n, cp in enumerate(cps):
        
    
_12mx = csvdf[(csvdf.subcat == subcat)]
_3mx = csvdf[(csvdf.coupling == cp) & (csvdf.subcat == subcat)]['3mx']
        #plt.bar(index, [_12mx, _3mx], bar_width, bottom=y_offset, color=colors[n])
        
fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(12,8))
for n,subcat in enumerate(subcats):
    dfsubcat = csvdf[(csvdf.subcat == subcat)]
    
    # combine nt and t for k=1+2
    lastbottom_t, lastbottom_nt = np.zeros(4), np.zeros(4)
    for kclus, kclustr in zip(['12mx', '3smx', '3dmx', '4mx'], ['k1+2', 'k3 sparse', 'k3 dense', 'k4+5']):
        nt_ = csvdf[(csvdf.subcat == subcat) & (csvdf.tnt == 'n')][kclus]
        t_ = csvdf[(csvdf.subcat == subcat) & (csvdf.tnt == 't')][kclus]
        tntsum_t = t_.values
        tntsum_nt = nt_.values
        ax[0,n].bar(cps, tntsum_nt, bottom=lastbottom_nt, label=kclustr) 
        ax[1,n].bar(cps, tntsum_t, bottom=lastbottom_t, label=kclustr) 
        
        lastbottom_t = lastbottom_t + tntsum_t
        lastbottom_nt = lastbottom_nt + tntsum_nt
    ax[0,n].set_xticklabels(labels=cps, rotation=45)
    ax[1,n].set_xticklabels(labels=cps, rotation=45)

    ax[0, n].set_title(f'nt {subcat}')
    ax[1,n].set_title(f't {subcat}')

    #ax[0, n].set_yticklabels([np.round(elem/2,2) for elem in np.arange(0,2.01,0.25)])
    #ax[1, n].set_yticklabels([np.round(elem/2,2) for elem in np.arange(0,2.01,0.25)])

    # proportion of k=1+2
    # ax[n].bar(cps, tntsum, bottom=0, label='k1+2') 
    # # proportion of k=3 dense
    # ax[n].bar(cps, dfsubcat['3dmx'].values, bottom=dfsubcat['12mx'], label='k3d') 
    # # proportion of k=3 sparse
    # ax[n].bar(cps, dfsubcat['3smx'].values, bottom=(dfsubcat['12mx'] + dfsubcat['3dmx']), label='k3s') 
    # # proportion of k=4 
    # ax[n].bar(cps, dfsubcat['4mx'].values, bottom=(dfsubcat['12mx'] + dfsubcat['3dmx'] + dfsubcat['3smx']), label='k4')     
    
    # ax[n].set_xticklabels(labels=cps, rotation=45)
    # ax[n].set_title(f'{subcat}')
    
ax[0,0].set_ylabel('proportion of trials')
ax[1,0].set_ylabel('proportion of trials')

handles, labels = ax[-1,-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='right')
#plt.tight_layout()
plt.suptitle('repeaters upgraders k cluster proportion of trials')
plt.savefig('./saved-plots/k-cluster-proportion-stacked-bar.png', dpi= 160)

#%%%% get k groups per subject cat for stats 


fi = open('./csvs/kgrps-4-takako.csv', 'w')

writer = csv.writer(fi) 
writer.writerow(['subject', 'subjectcat', 'coupling', 'kgrp'])

df = df[df['timbre'] == 'n']

for row, col in df.iterrows():
    kgrp = df.iloc[row]['k group']
    
    if isinstance(kgrp, str):
        kgrp = int(kgrp)+1
    elif np.isnan(kgrp):
        kgrp = '1+2'
    else:
        kgrp = int(kgrp)+1    

        
    subject = df.iloc[row]['subject']
    cp = df.iloc[row]['coupling']
    subcat = df.iloc[row]['subject cat']
   
    denseflag = df.iloc[row]['dispersion group']
    
    if isinstance(denseflag, str):
        kgrp = df.iloc[row]['dispersion group']
        kgrp = str(kgrp)
        print(kgrp)
    elif (kgrp == 1) or (kgrp == 2):
        kgrp = '1+2'
    elif kgrp == 4:
        kgrp = '4'
    elif kgrp == 5:
        kgrp = '5'
    writer.writerow([subject, subcat, cp, kgrp])
    

fi.close()
#%% sanity check 
# .125 -> PARTICIPANT_kura-A2_2020-09-11_12h55.12.335.csv
data = pd.read_csv('./csvs/kgrps-4-takako.csv')

subcat = 'regular'
cp = 'none'
cluster = '4'
cluster1 = '3d' 
cluster2 = '3s'
totaltrials = data[(data['subjectcat'] == subcat) & (data['coupling'] == cp) ].count()[0]

dfslice = data[(data['subjectcat'] == subcat) & (data['coupling'] == cp) & (data['kgrp'] == cluster)].count()[0]

# dfslice = data[(data['subjectcat'] == subcat) & (data['coupling'] == cp) & ((data['kgrp'] == cluster1) | (data['kgrp'] == cluster2))].count()[0]

print(dfslice/totaltrials)

