#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:48:15 2021

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

import random
import csv
import patsy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLMResults
import seaborn as sns
from fun.functions import *

#%% ### CHECK NUMBER OF SUBJECTS WHO SWITCHED CATEGORIES FROM NOTIMBRE TO TIMBRE 
timbrestr = 't' # 'n' or 't'
rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/' + timbrestr + '/all'
dft = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + timbrestr + '.pkl'))

# exclude PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv
dft = dft[dft['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv']
dft.reset_index(inplace=True)

timbrestr = 'nt'
rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/' + timbrestr + '/all'
dfnt = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + timbrestr + '.pkl'))
dfnt = dfnt[dfnt['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv']
dfnt.reset_index(inplace=True)


unique_subjects = list(set(dft['subject']))

tsubjcat = {dft['subject'][i]: dft['subject cat'][i] for i in range(len(dft['subject']))}
#tsubjcat = {dft['subject'][i]: dft['subject cat'][i] for i in range(len(dft['subject']))}

ntsubjcat = {dfnt['subject'][i]: dfnt['subject cat'][i] for i in range(len(dfnt['subject']))}

f = open('./mturk-csv/usable-batch-12-7/subject-raster-plots/' + 'sub cat comparison.csv', 'w')
writer = csv.writer(f)
writer.writerow(['subject', 'nt', 't'])

switchers = {'subject': [], 'block1':[], 'block2':[]}

num = 0
for subject in unique_subjects:
    writer.writerow([subject,ntsubjcat[subject], tsubjcat[subject]])
    
    if ntsubjcat[subject] != tsubjcat[subject]:
        #print(f'{subject} switched')
        print(subject, ntsubjcat[subject], tsubjcat[subject])

        num+=1 
print(f'\n{num} subjects switched categories between nt,t')
f.close()

## total = 15 switchers 

#%%
# set root dir
timbrestr = 'nt'
#timbrestr = 't'

rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/' + timbrestr + '/all'

# read most up to date generated .pkl file in ./csvs/ 
df = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + timbrestr + '.pkl'))

df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]


#%%

# how many loyalists 
theloyalists = list(set(df[df['subject cat'] == 'loyalist']['subject']))
theconverts = list(set(df[df['subject cat'] == 'convert']['subject']))
thenoneconverts = list(set(df[df['subject cat'] == 'noneconvert']['subject']))

numloyalist, numnoneconverts, numconverts = len(theloyalists), len(thenoneconverts), len(theconverts)

print(f'number loyalist: {numloyalist} \nnum noneconverts: {numnoneconverts} \nnum converts: {numconverts}')

#
##################################################################################
#%% ############ ####KDE histogram ASYNCHRONIES ####################################
##################################################################################
## look at beat per beat point plot for both t and nt



numbeats = 19
cps = ['strong', 'medium', 'weak', 'none']
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)]


from scipy import stats


fig_i, ax_i = plt.subplots(nrows=4, ncols=3, figsize=(20,8), sharex=True, sharey=True)

for k, ts in enumerate(['nt']):
    
    
    
    rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/' + ts + '/all'
    df = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + ts + '.pkl'))
    df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv']
    df.reset_index()
  
    mxitipools, sxitipools = {}, {}
    for i, subcat in enumerate(['loyalist', 'noneconvert', 'convert']):
        for n, cp in enumerate(cps):
            mxs, sxs = [], []
            mxitipools[cp], sxitipools[cp] = [], []
            for person in sorted(list(set(df[df['subject cat'] == subcat]['subject']))): 
                mxtmp = df[(df['coupling'] == cp) & (df['subject'] == person)]['subjects norm itis']
                for arr in mxtmp:
                    mxitipools[cp].extend(arr[:numbeats])

            xbins = np.linspace(0, 1.5, 100)            
 
            color_ = sns.color_palette()[n]

            ax_i[n,i].hist(mxitipools[cp], bins=xbins, alpha=0.5, color=color_)        
            kde_sync_cond = stats.gaussian_kde(mxitipools[cp])  
            ax_i[n,i].plot(xbins, kde_sync_cond(xbins))
            ax_i[n,i].set_xlim([0,1.5])
            ax_i[n,i].set_title(f' {cp}')
            
            nITI_mx = np.round(np.mean(mxitipools[cp]),2)
            nITI_sx = np.round(np.std(mxitipools[cp]),2)
            print(f'{subcat} {cp} : mx: {nITI_mx} sx: {nITI_sx}')


        #Npercat = len(list(set(df[df['subject cat'] == subcat]['subject']))) # number of subjects in category
        ##### PLOT HISTOGRAM OF AGGREGATED ASYNCHRONIES PER COUPLING AND SUBJECT CAT            
        # xbins = np.linspace(0, 1.5, 100)
        # # ax_i[i,k].hist(mxitipools['none'], bins=xbins, alpha=0.5, label='none')
        # # ax_i[i,k].hist(mxitipools['weak'], bins=xbins, alpha=0.5, label='weak')
        # # ax_i[i,k].hist(mxitipools['medium'], bins=xbins, alpha=0.5, label='medium')
        # # ax_i[i,k].hist(mxitipools['strong'], bins=xbins, alpha=0.5, label='strong')
 
        # # KDE? 
        # kden = stats.gaussian_kde(mxitipools['none'])
        # kdew = stats.gaussian_kde(mxitipools['weak'])
        # kdem = stats.gaussian_kde(mxitipools['medium'])
        # kdes = stats.gaussian_kde(mxitipools['strong'])

        # ax_i[i,k].plot(xbins, kden(xbins), label='none')
        # ax_i[i,k].plot(xbins, kdew(xbins), label='weak')
        # ax_i[i,k].plot(xbins, kdem(xbins), label='medium')
        # ax_i[i,k].plot(xbins, kdes(xbins), label='strong')

        # ax_i[i,k].set_xlim([0,1.5])
        #ax_i[i,k].set_title(f'{ts} {subcat} N={Npercat}')
        #ax_i[i,k].legend()

plt.legend(title='coupling', bbox_to_anchor=(1.05, 1), loc='upper left')

#fig_i.suptitle(f'asynchronies per subject category and coupling condition')
for ax_ in ax_i[-1,:].flat:
    ax_.set_xlabel('normalized ITI')
    #ax_.set_ylabel('tap count')

# ax_i[0,0].set_title('Regular Tappers')
# ax_i[0,1].set_title('Fast Tappers')
# ax_i[0,2].set_title('Hybrid Tappers')



# fig_i.savefig(os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/histograms/', f'asynchronies distribution per subcat 4x3.eps'))

#fig_i.savefig(os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/histograms/', f'asynchronies distribution per subcat 4x3.png'), dpi=120)

#%% NB:: dont need to run, this is jus subjects who didn't switch subcats between nt and t 
####### THIS IS LOYAL LOYALIST, LOYAL NC, LOYAL C....
##### e.g. LOOK AT SUBJECTS WHO DIDN"T SWITCH TAPPING STRATEGY BETWEEN TIMBRE BLOCKS !!!!
######## these are subjects who didn't switch between t and nt 
loyal_loyalists = ['PARTICIPANT_kura-A2_2020-09-16_17h20.10.296.csv',
 'PARTICIPANT_kura-A1_2020-10-04_12h32.21.589.csv',
 'PARTICIPANT_kura-A2_2020-11-25_15h04.36.125.csv',
 'PARTICIPANT_kura-A1_2020-10-03_10h29.30.435.csv',
 'PARTICIPANT_kura-A1_2020-10-01_10h17.21.641.csv',
 'PARTICIPANT_kura-B2_2020-09-25_19h37.04.527.csv',
 'PARTICIPANT_kura-B2_2020-09-10_18h03.12.198.csv',
 'PARTICIPANT_kura-A2_2020-09-16_10h16.41.575.csv',
 'PARTICIPANT_kura-A1_2020-10-05_06h37.13.057.csv',
 'PARTICIPANT_kura-B1_2020-10-05_07h49.31.729.csv',
 'PARTICIPANT_kura-A1_2020-10-05_09h41.04.130.csv',
 'PARTICIPANT_kura-A2_2020-11-05_09h19.11.832.csv',
 'PARTICIPANT_kura-B2_2020-09-07_12h08.36.208.csv',
 'PARTICIPANT_kura-A1_2020-10-05_06h11.14.448.csv',
 'PARTICIPANT_kura-A2_2020-11-04_17h32.29.185.csv',
 'PARTICIPANT_kura-B2_2020-09-07_21h13.50.539.csv']
loyal_noneconverts = ['PARTICIPANT_kura-B2_2020-09-16_13h54.10.653.csv',
 'PARTICIPANT_kura-A2_2020-08-28_09h24.01.653.csv',
 'PARTICIPANT_kura-B2_2020-09-13_16h50.25.586.csv',
 'PARTICIPANT_kura-A1_2020-10-01_08h10.36.781.csv',
 'PARTICIPANT_kura-B1_2020-10-05_08h31.10.554.csv',
 'PARTICIPANT_kura-A2_2020-09-16_12h21.37.513.csv',
 'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv',
 'PARTICIPANT_kura-A2_2020-11-06_20h46.36.344.csv',
 'PARTICIPANT_kura-A2_2020-09-16_12h19.57.327.csv',
 'PARTICIPANT_kura-B2_2020-09-16_12h17.20.371.csv',
 'PARTICIPANT_kura-A2_2020-09-07_10h21.43.606.csv',
 'PARTICIPANT_kura-A2_2020-09-16_13h15.20.738.csv']
loyal_converts = ['PARTICIPANT_kura-A2_2020-09-16_15h15.27.472.csv',
 'PARTICIPANT_kura-A1_2020-10-05_11h28.35.951.csv',
 'PARTICIPANT_kura-A1_2020-10-05_06h11.28.743.csv',
 'PARTICIPANT_kura-B1_2020-10-05_10h39.48.518.csv',
 'PARTICIPANT_kura-A2_2020-09-07_10h52.13.829.csv',
 'PARTICIPANT_kura-B2_2020-09-07_10h04.17.709.csv',
 'PARTICIPANT_kura-B1_2020-09-16_15h23.50.906.csv']

cps = ['none', 'weak', 'medium', 'strong']
numbeats = 19 
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)]

#### make csv for comapring loyal loyalist, loyal converts, loyal noneconverts 
rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/csvs/'

df = pd.read_csv(os.path.join(rootdir, 'all-stats-4-7-tnt.csv'))
df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv'] #remove this subject
df.reset_index()


for batch_str, subject_batch in zip(['loyal-l', 'loyal-nc', 'loyal-c'], [loyal_loyalists, loyal_noneconverts, loyal_converts]):
    #print(batch_str, subject_batch)
    f = open(os.path.join(rootdir, batch_str + '.csv'), 'w')
    writer = csv.writer(f)
    writer.writerow(['subject', 'timbre', 'coupling', 'beatsection', 'mx', 'sx']) 
    
    for person in subject_batch:    
        for k, ts in enumerate(['n', 't']):
            for n, cp in enumerate(cps):
                mxs, sxs = [], []
                tapsmx, tapssd = np.empty(numbeats), np.empty(numbeats)
                
                for bt in range(numbeats):
                    dt = df[(df['coupling'] == cp) & (df['subject'] == person) & (df['timbre'] == ts)][str(bt)]
                    mxtaps = np.nanmean(dt)
                    sxtaps = np.nanstd(dt)
                    tapsmx[bt] = mxtaps
                    tapssd[bt] = sxtaps
        
                mxtapsections, sxtapsections = [], []
                for b, bseg in enumerate(beatsections):
                    tapsegmx = np.nanmean(tapsmx[bseg[0]:bseg[1]])
                    mxtapsections.append(tapsegmx)
                    
                    tapsegsx = np.nanmean(tapssd[bseg[0]:bseg[1]])
                    sxtapsections.append(tapsegsx)
                
                    writer.writerow([person, ts, cp, b, tapsegmx, tapsegsx])
    f.close()   
############################
### impute means from just generated csvs (for R)

# for batch_str in ['loyal-l', 'loyal-nc', 'loyal-c']:
    
#     df = pd.read_csv('./mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/csvs/' + batch_str + '.csv')        
#     df_imputed = impute_mean(df, 'mx')
#     df_imputed = impute_mean(df_imputed, 'sx')
    
#     df_imputed.to_csv('./mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/csvs/' + batch_str + '-imp.csv')
#%%

# 222 total 3d + 3s in hybrid 
# how many are 3d? 153/222 = 69%
# how many are 3s? 69/222 = 31%

#%%

rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/'
df = pd.read_csv(os.path.join(rootdir, 'csvs/all-stats-4-7-tnt.csv'))
df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv'] #remove this subject
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]

df.reset_index() 

n_per_cat = getNumberSubjectsPerCat(df)

#fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8), sharex=True, sharey=False)

numbeats = 19 
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)]

# ITIs for t+nt LOYALS 
# ITIs for t, nt LOYALS
cplabels = ['loyal loyalist', 'loyal noneconvert', 'loyal convert']

for k, ts in enumerate(['t','n', "_"]):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8), sharex=True, sharey=False)

    for subcatstr, subbatch in zip(cplabels, [loyal_loyalists, loyal_noneconverts, loyal_converts]):
        f = open(os.path.join(rootdir, 'csvs', subcatstr + '.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(['subject', 'coupling', 'beatsection', 'mx', 'sx'])
        
            
        for n, cp in enumerate(cps):
            mxs, sxs = [], []
    
            for person in subbatch:
        
                tapsmx, tapssd = np.empty(numbeats), np.empty(numbeats)
                for bt in range(numbeats):
                    if k < 2:
                        dt = df[(df['coupling'] == cp) & (df['subject'] == person) & (df['timbre'] == ts)][str(bt)]
                    else:
                        dt = df[(df['coupling'] == cp) & (df['subject'] == person)][str(bt)]
                        ts = 't+nt'
                    mxtaps = np.nanmean(dt)
                    sxtaps = np.nanstd(dt)
                    tapsmx[bt] = mxtaps
                    tapssd[bt] = sxtaps
                #mx = np.nanmean(taps)
                #sx = np.nanstd(taps)
                
                mxtapsections, sxtapsections = [], []
                for b, bseg in enumerate(beatsections):
                    tapsegmx = np.nanmean(tapsmx[bseg[0]:bseg[1]])
                    mxtapsections.append(tapsegmx)
                    
                    tapsegsx = np.nanmean(tapssd[bseg[0]:bseg[1]])
                    sxtapsections.append(tapsegsx)
                
                    writer.writerow([person, cp, b, tapsegmx, tapsegsx])
                
                mxs.append(np.array(mxtapsections))
                sxs.append(np.array(sxtapsections))
            
            mxs = np.array(mxs) 
            meanmxs = np.nanmean(mxs, axis=0)
            mxerror = np.nanstd(mxs, axis=0)
    
            sxs = np.array(sxs)
            meansxs = np.nanmean(sxs, axis=0)
            sxerror = np.nanstd(meansxs, axis=0)
                
            xr = np.arange(0+n/7, len(beatsections)+n/7, 1)
            ax[0,i].errorbar(x=xr, y=meanmxs, yerr=mxerror, label=cp) 
            #ax[0,i].legend()
            ax[0,i].set_title(f'{cplabels[i]} N={len(subbatch)}')
            ax[0,i].set_xticks(np.arange(len(beatsections)))
            ax[0,i].set_xticklabels([str(elem+1) for elem in range(len(beatsections))])
            ax[0,i].set_ylabel('norm. ITI')
    
            ax[1,i].errorbar(x=xr, y=meansxs, yerr=sxerror, label=cp) 
            #ax[1,i].legend()
            ax[1,i].set_title(f'{cplabels[i]}')
            ax[1,i].set_xticks(np.arange(len(beatsections)))
            ax[1,i].set_xticklabels([str(elem+1) for elem in range(len(beatsections))])
            ax[1,i].set_ylabel('norm. STD')
            
            ax[0, i].set_ylim([0, 1.8])
            ax[1, i].set_ylim([0, 1.0])
            
    
            
    
        f.close()
            
        for ax_ in ax[1,:].flat:
            ax_.set_xlabel('subject Nth segment tap')
        
    handles, labels = ax[0,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'{ts} nITI between subject groups: {len(beatsections)} segments, each with {lenbeatsection} beats')
        
    #fig.savefig(os.path.join(rootdir,'iti-plots', f'loyal sub cat t+nt iti.eps'), dpi=120)
    #fig.savefig(os.path.join(rootdir,'iti-plots', f'loyal sub cat t+nt iti.png'), dpi=120)

    fig.savefig(os.path.join(rootdir,'iti-plots', f'{ts} loyal sub cat iti.png'), dpi=120)


#%%#################################################################
############# ITI PER BEAT SEGMENT AND SUBJECT CATEGORY #############
##############################################################################
cps = ['strong', 'medium', 'weak', 'none']
numbeats = 19 
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)]



# initialize dict of sx curves
sxcurves = {}
for s_ in ['nt', 't']:
    sxcurves[s_] = {}
    for subcat in ['loyalist', 'noneconvert', 'convert']:
        sxcurves[s_][subcat] = {}
        for cp in cps:
            sxcurves[s_][subcat][cp] = 0
    
for k, ts in enumerate(['t', 'nt']):
    
    rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/' + ts + '/all'
    df = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + ts + '.pkl'))
    #df = pd.read_pickle(os.path.join(rootdir, 'csvs', 'all-stats-4-7-' + ts + '.pkl'))


    #df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv'] #remove this subject
    df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
    df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]
    df.reset_index()    
    
    n_per_cat = getNumberSubjectsPerCat(df)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8), sharex=True, sharey='row')

    fagg = open(os.path.join(rootdir, 'csvs', ts + '.csv'), 'w')
    writer2 = csv.writer(fagg)
    writer2.writerow(['subcat', 'coupling', 'tap section', 'mx', 'sx'])
    
    f3 = open(os.path.join(rootdir, 'csvs', 'all-rev.csv'), 'w')
    writer3 = csv.writer(f3)
    writer3.writerow(['subject', 'subcat', 'coupling', 'tapsection', 'mx', 'sx'])   
    # hold sx curves for further analysis (curve fitting)   

    # make ITIs per tap strat cat 
    for i, subcat in enumerate(['loyalist', 'noneconvert', 'convert']):
            
        f = open(os.path.join(rootdir, 'csvs', subcat + '.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(['subject', 'coupling', 'beatsection', 'mx', 'sx'])

        
        for n, cp in enumerate(cps):
            mxs, sxs = [], []
            for person in sorted(list(set(df[df['subject cat'] == subcat]['subject']))):
                    
                # get nth beat/numbeats from each one of a subject's trials for each coupling condition (10 total). append to an array  
                
                # tapsmx, tapssd = np.empty(numbeats), np.empty(numbeats)
                # for bt in range(numbeats):
                #     dt = df[(df['coupling'] == cp) & (df['subject'] == person)][str(bt)]
                #     mxtaps = np.nanmean(dt)
                #     sxtaps = np.nanmean(dt)
                #     tapsmx[bt] = mxtaps
                #     tapssd[bt] = sxtaps
                    
                
                mxtapsections, sxtapsections = [], []
                for b, bseg in enumerate(beatsections):
                    brange = [str(num) for num in np.arange(bseg[0], bseg[1])] # get columns for beats n to n+3 for all of one subjects tap responses for a coupling cond 
                    try:
                        dt = df[(df['coupling'] == cp) & (df['subject'] == person)][brange]
                        mxsectmx = np.nanmean(dt)
                        sxsectsx = np.nanstd(dt)
                    except:
                        try:
                            newbrange = brange[:-1]
                            dt = df[(df['coupling'] == cp) & (df['subject'] == person)][newbrange]
                            mxsectmx = np.nanmean(dt)
                            sxsectsx = np.nanstd(dt)
                        except:
                            newbrange = newbrange[:-1]
                            dt = df[(df['coupling'] == cp) & (df['subject'] == person)][newbrange]
                            mxsectmx = np.nanmean(dt)
                            sxsectsx = np.nanstd(dt)                            



                    # mxsectmx = np.nanmean(mxsectmx)
                    # sxsectsx = np.nanstd(mxsectmx)
                    
                    mxtapsections.append(mxsectmx)
                    sxtapsections.append(sxsectsx)
                    
                    writer.writerow([person, cp, b, mxsectmx, sxsectsx])
                    writer3.writerow([person, subcat, cp, b, mxsectmx, sxsectsx])                    
                
                
                # mxtapsections, sxtapsections = [], []
                # for b, bseg in enumerate(beatsections):
                #     tapsegmx = np.nanmean(tapsmx[bseg[0]:bseg[1]])
                #     mxtapsections.append(tapsegmx)

                #     tapsegsx = np.nanstd(tapssd[bseg[0]:bseg[1]])
                #     sxtapsections.append(tapsegsx)
                    
                #     # tapsegsx = np.nanmean(tapssd[bseg[0]:bseg[1]])
                #     # sxtapsections.append(tapsegsx)
                
                #     writer.writerow([person, cp, b, tapsegmx, tapsegsx])
                #     writer3.writerow([person, subcat, cp, b, tapsegmx, tapsegsx])
                    
                    
                    
                # vertically stack mx,sx sections per each subject    
                mxs.append(np.array(mxtapsections))
                sxs.append(np.array(sxtapsections))
            
            # get between subject SEM error    
            mxs = np.array(mxs) 
            meanmxs = np.nanmean(mxs, axis=0)
            mxerror = np.nanstd(mxs, axis=0)
    
            sxs = np.array(sxs)
            meansxs = np.nanmean(sxs, axis=0)
            sxerror = np.nanstd(meansxs, axis=0)
            
            # save sx curves
            sxcurves[ts][subcat][cp] = meansxs
            for b, bseg in enumerate(beatsections):
                
                writer2.writerow([subcat, cp, b, meanmxs[b], meansxs[b]])
                
            xr = np.arange(0+n/7, len(beatsections)+n/7, 1)
            ax[0,i].errorbar(x=xr, y=meanmxs, yerr=mxerror, label=cp, capsize=4, marker='o') 
            #ax[0,i].legend()
            ax[0,i].set_title(f'{subcat}')
            ax[0,i].set_xticks(np.arange(len(beatsections)))
            ax[0,i].set_xticklabels([str(elem+1) for elem in range(len(beatsections))])
            ax[0,i].set_ylabel('norm. ITI')
    
            # log y scale? 
            # ax[0,i].set_yscale('log')
            # ax[1,i].set_yscale('log')
            
            # mean rows
            yticks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            ax[0,i].set_yticks(yticks)
            ax[0,i].set_yticklabels(yticks)
            
            ax[1,i].set_yticks([0.25, 0.5, 0.75, 1.0])
            ax[1,i].set_yticklabels([0.00, 0.25, 0.5, 1.0])
            
            # ax[1,i].set_yticks([0.00,0.25])
            # ax[1,i].set_yticklabels([0.00, 0.25])
            
            ax[1,i].errorbar(x=xr, y=meansxs, yerr=sxerror, label=cp, capsize=4, marker='o') 
            #ax[1,i].legend()
            ax[1,i].set_title(f'{subcat}')
            ax[1,i].set_xticks(np.arange(len(beatsections)))
            ax[1,i].set_xticklabels([str(elem+1) for elem in range(len(beatsections))])
            ax[1,i].set_ylabel('norm. STD')
            
            
            ax[0, i].set_ylim([0, 2.0])
            ax[1, i].set_ylim([0, 1.2])
            
            #ax[1, i].set_ylim([0, 0.28])
            

        f.close()
            
        for ax_ in ax[1,:].flat:
            ax_.set_xlabel('Tap Section')
            
    handles, labels = ax[0,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.suptitle(f'{ts} nITI between subject groups: {len(beatsections)} segments, each with {lenbeatsection} beats')
  
f.close()
fagg.close()
f3.close()
# save normal plots      
# fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg.eps'), dpi=120)
# fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg.png'), dpi=120)

### save log plots
    # fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg-log.eps'), dpi=120)
    # fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'loyal tap categories-beat-seg-log.png'), dpi=120)
#%%  CURVE FITTING TO DECAYING EXPONENTIAL 
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

#%%##############################
##### curve fitting for sx over tap section per tapping group ######
################################ 
import scipy as sp
import scipy.optimize
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a*np.exp(b*x) + c

sns.color_palette("tab10")

# initialize dict of sx curves
axcurves, curveparams = {}, {}
for ts in ['nt', 't']:
    axcurves[ts], curveparams[ts] = {}, {}
    for subcat in ['loyalist', 'noneconvert', 'convert']:
        axcurves[ts][subcat], curveparams[ts][subcat] = {}, {}
        for cp in cps:
            axcurves[ts][subcat][cp], curveparams[ts][subcat][cp] = 0, 0



t = np.arange(1,8)  # dep var, x-axis tap section vector  
for k, ts in enumerate(['nt']): ### NB: only for nt, can change to t to do the same 
    csvdir = os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/', ts, 'all', 'csvs/curve-fitting')
    fi = open(os.path.join(csvdir, ts + '-params-R.csv'), 'w')
    writer = csv.writer(fi)
    writer.writerow(['group', 'coupling', 'A', 'B', 'R2'])
    
    fig, ax = plt.subplots(nrows = 1, ncols = 3, sharex=True, sharey=True, figsize=(20,4))
    for s, subcat in enumerate(['loyalist','noneconvert', 'convert']):
        for n, cp in enumerate(cps):
            print(f'working on {ts} {subcat} {cp}')
            C0 = sxcurves[ts][subcat][cp][0]
            curve = sxcurves[ts][subcat][cp]
            #A, K = fit_exp_linear(t, curve, C0)
            #A, K, C = fit_exp_nonlinear(t, curve)
            #axcurves[ts][subcat][cp] = (A,K)
            # plot it
            #y = model_func(t, A, K, C0)
            
            # method 1
            ## NB: 
            '''   
            For fitting y = AeBx, take the logarithm of both side gives log y = log A + Bx. So fit (log y) against x.

Note that fitting (log y) as if it is linear will emphasize small values of y, causing large deviation for large y. 
            '''
            # z = scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  t,  curve)   
            # print(f'\t {np.round(z[0][0],2)}*np.exp({np.round(z[0][1],3)}*t)')
            # ax[s].plot(t, z[0][0]*np.exp(z[0][1]*t), label=cp, marker='x', color=sns.color_palette('tab10')[n])              
 
            # method 2, 
            # NB: For y = AeBx, however, we can get a better fit since it computes Δ(log y) directly.
            z = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  t,  curve)  
            print(f'\t {np.round(z[0][0],2)}*np.exp({np.round(z[0][1],3)}*t)')
            app_curve = z[0][0]*np.exp(z[0][1]*t)
            real_curve = np.copy(sxcurves[ts][subcat][cp])
            r2 = compute_gof(real_curve, app_curve)
            print(f'{ts} {subcat} {cp} : {r2}')
            
            ax[s].plot(t, app_curve, label=cp, marker='x', color=sns.color_palette('tab10')[n]) 
            curveparams[ts][subcat][cp] = z[0]
            
            
            ## method 3
            # popt, pcov = curve_fit(func, t, curve, maxfev=5000)
            # ax[s].plot(t, func(t,popt[0], popt[1], popt[2]), label=cp, marker='x')
          
            ax[s].plot(t, curve, color=sns.color_palette('tab10')[n], linewidth=0.5)
            ax[s].set_xlabel('Tap Section')
            ax[s].set_title(subcat) 
            
            
            # write params to csv file 
            writer.writerow([subcat, cp, z[0][0], z[0][1], r2])
            
    plt.legend()
    #plt.savefig(os.path.join(csvdir, ts + '-params.png'), dpi=120)
    plt.savefig('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/curve-fit-rev.png', dpi=120)
    
fi.close()

#%% PLOT THE A and B COEFFICETS AGAINST R stim 

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10,4))
ax[0].set_title('A coefficients')
ax[1].set_title('B coefficients')
ax[0].set_xlabel('|R| stimulus')
ax[1].set_xlabel('|R| stimulus')

df = pd.read_csv(os.path.join(csvdir, ts + '-params-R.csv'))

#Rs = {'strong' : 0.97, 'medium': 0.46, 'weak':0.35, 'none': 0.09}
Rs = [0.97, 0.46, 0.32, 0.09]

for k, ts in enumerate(['nt']):
    for i, coeff in enumerate(['A', 'B']):
        for subcat_, subcat in zip(['regular', 'hybrid', 'fast'], ['loyalist', 'noneconvert', 'convert']):
            A = df[df['group'] == subcat][coeff].values
            #r_stim = df[df['group'] == subcat]['R_stim'].values
            
            ax[i].plot(Rs, A, label=subcat_, marker='.')
        
                        
        handles, labels = ax[i].get_legend_handles_labels()
        #fig.legend(handles, labels, loc='right')

plt.legend(title='tapping group', bbox_to_anchor=(1.5, 1), loc='center right')
# plt.savefig(os.path.join(csvdir, 'A-B-coeffs.png'), bbox_inches = 'tight', dpi=130)
# plt.savefig(os.path.join(csvdir, 'A-B-coeffs.eps'), bbox_inches = 'tight')

plt.tight_layout()
plt.savefig('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/curve-fit-coeffs.png', dpi=120)

            

#%%%
t = np.arange(1,8)
z = scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  t,  sxcurves['nt']['loyalist']['none'])
#%% get average |R| stim for NT stimuli 
# for nt, just get unique_stims from one subject from b1 and one from b2, doesn't matter who 
b1stims = list(dfnt[dfnt['subject'] == 'PARTICIPANT_kura-B1_2020-09-10_21h13.09.622.csv']['stim']) 
b2stims = list(dfnt[dfnt['subject'] == 'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv']['stim']) 
nt_unique_stims = b1stims + b2stims

nt_R_model = {}
for stim in b1stims:
    nt_R_model[stim] = dfnt[(dfnt['subject'] == 'PARTICIPANT_kura-B1_2020-09-10_21h13.09.622.csv') & (dfnt['stim'] == stim)]['R model']
for stim in b2stims:
    nt_R_model[stim] = dfnt[(dfnt['subject'] == 'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv') & (dfnt['stim'] == stim)]['R model']
    

R_s = {'strong': [], 'medium': [], 'weak': [], 'none': []}
for stim in nt_unique_stims:  
    for s_ in ['strong','medium','weak','none']:
        if stim.startswith(s_):
            R_s[s_].append(nt_R_model[stim])

print(f'... for no-timbre stimuli, phase coherence avg:')    
for s_ in ['strong', 'medium', 'weak', 'none']:
    R_mx = np.mean(R_s[s_])
    print(f'R mean {s_}: {R_mx}')
print('\n\n')

##############################    
# same thing but for 't', get average |R| for T stim
b1stims = list(dft[dft['subject'] == 'PARTICIPANT_kura-B1_2020-09-10_21h13.09.622.csv']['stim']) 
b2stims = list(dft[dft['subject'] == 'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv']['stim']) 
t_unique_stims = b1stims + b2stims

t_R_model = {}
for stim in b1stims:
    t_R_model[stim] = dft[(dft['subject'] == 'PARTICIPANT_kura-B1_2020-09-10_21h13.09.622.csv') & (dft['stim'] == stim)]['R model']
for stim in b2stims:
    t_R_model[stim] = dft[(dft['subject'] == 'PARTICIPANT_kura-B2_2020-09-16_12h31.45.449.csv') & (dft['stim'] == stim)]['R model']
    

R_s = {'strong': [], 'medium': [], 'weak': [], 'none': []}
for stim in t_unique_stims:  
    for s_ in ['strong','medium','weak','none']:
        if stim.startswith(s_):
            R_s[s_].append(t_R_model[stim])
            
print(f'... for timbre stimuli, phase coherence avg:')
for s_ in ['strong', 'medium', 'weak', 'none']:
    R_mx = np.mean(R_s[s_])
    print(f'R mean {s_}: {R_mx}')


#%% plot R stim vs. approximated curve params
R_model = {'strong': 0.972, 'medium': 0.461, 'weak': 0.346, 'none': 0.0818}

fig, ax = plt.subplots(nrows = 2, ncols=3, sharex=True, sharey=True, figsize=(10,6))
params = {'mathtext.default': 'regular' }          

for ts in ['nt']: # NB: this is just 'nt', need to do 't' for timbre paper 
    for n, subcat in enumerate(['loyalist', 'noneconvert', 'convert']):
        a_vect, b_vect = [], []
        for cp in cps:
            a_vect.append(curveparams[ts][subcat][cp][0])
            b_vect.append(curveparams[ts][subcat][cp][1])
        R_avg_vect = [R_model[val] for val in R_model]
        ax[0, n].plot(a_vect, label='a coeffs', marker='v')
        ax[0, n].plot(R_avg_vect, marker = 'o', label='|R| stim avg')
        
        ax[1, n].plot(b_vect, label='b coeffs', marker='^')
        ax[1, n].plot(R_avg_vect, marker = 'o', label='|R| stim avg')
        
        ax[0, n].set_xticks([0,1,2,3])
        ax[0, n].set_xticklabels(['strong','medium', 'weak','none'])
        
        ax[0, n].set_title(f'{subcat} ')
        
    
    ax[0, 0].set_ylabel('$|R|_{stim}$ / A coeff')
    ax[1, 0].set_ylabel('$|R|_{stim}$ / B coeff')


    ax[0, 2].legend()
    ax[1, 2].legend()

#plt.savefig(os.path.join(csvdir, 'Rstim_vs_params.png'), dpi=120)

            

#%% impute means

for i, subcat in enumerate(['loyalist', 'noneconvert', 'convert']):
    
    df_ = pd.read_csv(os.path.join(rootdir, 'csvs', subcat + '.csv'))
    df_imputed = impute_mean(df_, 'mx')
    df_imputed = impute_mean(df_imputed, 'sx')

    
    df_.to_csv(os.path.join(rootdir, 'csvs', subcat + '-imp.csv'))
    
#%% 
df_ = pd.read_csv(os.path.join(rootdir, 'csvs', 'all-rev.csv'))
imp_sx = np.nanmean(df_['sx'].values)
imp_mx = np.nanmean(df_['mx'].values)

df_['sx'].fillna(imp_sx, inplace=True)
df_['mx'].fillna(imp_mx, inplace=True)


df_.to_csv(os.path.join('./mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all', 'csvs', 'all-rev-imp.csv'))






