#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:48:08 2023

@author: nolanlem
"""

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

import load_subjects_resps 

os.chdir('/Users/nolanlem/Desktop/data/Experiment-2/scripts/')

# load up the subject respsonses from the other script 
#subject_resps, sndbeatbins = load_subjects_resps.load_subjects_resps()


#%%

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

#%%
df = pd.read_csv('./df-bb-taps-3-2.csv')  
# remove csvs
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]  



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
    #plt.savefig(os.path.join('./plots', f'K-mean-{nclusters}-clusters.png'),dpi=120)
    
    #finalsdir = '/Users/nolanlem/Documents/kura/swarmgen/nt-paper/images/in-paper/'
    #plt.savefig(os.path.join(finalsdir, f'K-mean-{nclusters}-clusters.eps'))
    #plt.savefig(os.path.join(finalsdir,f'K-mean-{nclusters}-clusters.tif'),dpi=120)

    return points, y_km

#%%



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
            

# save fig?
plt.savefig('./plots/k-means-50.png', dpi=150)

#%%
###########################################################################
############################# TIME COURSE nITI ##############################################
###########################################################################

# read in df
df = pd.read_csv('./all-stats-4-7-nt-new.csv');

df['subject cat'] = df['subject cat'].replace({'loyalist':'regular', 'noneconvert':'hybrid', 'convert':'fast'})


df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]  



#################################################################
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
    for subcat in ['regular', 'hybrid', 'fast']:
        sxcurves[s_][subcat] = {}
        for cp in cps:
            sxcurves[s_][subcat][cp] = 0
    
for k, ts in enumerate(['nt']):
    
        
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8), sharex=True, sharey='row')

    # csv files for mx and sx over tap section
    fagg = open(os.path.join('./csvs', ts + '.csv'), 'w')
    writer2 = csv.writer(fagg)
    writer2.writerow(['subcat', 'coupling', 'tap section', 'sx'])
    
    f3 = open(os.path.join('./csvs', 'all-rev.csv'), 'w')
    writer3 = csv.writer(f3)
    writer3.writerow(['subject', 'subcat', 'coupling', 'tapsection', 'mx', 'sx'])   
    # hold sx curves for further analysis (curve fitting)   

    # make ITIs per tap strat cat 
    for i, subcat in enumerate(['regular', 'hybrid', 'fast']):
            
        f = open(os.path.join('csvs', subcat + '.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(['subject', 'coupling', 'beatsection', 'mx', 'sx'])
       
        for n, cp in enumerate(cps):
            mxs, sxs = [], []
            for person in sorted(list(set(df[df['subject cat'] == subcat]['subject']))):
                    

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

                    mxtapsections.append(mxsectmx)
                    sxtapsections.append(sxsectsx)
                    
                    writer.writerow([person, cp, b, mxsectmx, sxsectsx])
                    writer3.writerow([person, subcat, cp, b, mxsectmx, sxsectsx])                    

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
                
                writer2.writerow([subcat, cp, b, meanmxs[b]])
                
            xr = np.arange(0+n/7, len(beatsections)+n/7, 1)
            ax[0,i].errorbar(x=xr, y=meanmxs, yerr=mxerror, label=cp, capsize=4, marker='o') 
            #ax[0,i].legend()
            ax[0,i].set_title(f'{subcat} ')
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

            yticks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            
            ax[1,i].set_yticks(yticks)
            ax[1,i].set_yticklabels(yticks)

            ax[1,i].errorbar(x=xr, y=meansxs, yerr=sxerror, label=cp, capsize=4, marker='o') 

            ax[1,i].set_xticks(np.arange(len(beatsections)))
            ax[1,i].set_xticklabels([str(elem+1) for elem in range(len(beatsections))])
            ax[1,i].set_ylabel('norm. STD')
            
            
            ax[0, i].set_ylim([0, 2.0])
            ax[1, i].set_ylim([0, 1.2])

            

        f.close()
            
        for ax_ in ax[1,:].flat:
            ax_.set_xlabel('Tap Section')
            
    handles, labels = ax[0,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    #fig.suptitle(f'{ts} nITI between subject groups: {len(beatsections)} segments, each with {lenbeatsection} beats')
  
f.close()
fagg.close()
f3.close()

# # save normal plots      
fig.savefig(os.path.join('./plots/', 'time-course-nITI.eps'), dpi=150,bbox_inches='tight')
fig.savefig(os.path.join('./plots/', 'time-course-nITI.png'), dpi=150,bbox_inches='tight')


### save log plots
#fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg-log.eps'), dpi=120)
#fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'loyal tap categories-beat-seg-log.png'), dpi=120)

#%%

#################### PHASE COHERENCE PLOTS ########################
#################### PHASE COHERENCE PLOTS ########################

df = pd.read_csv('./df-bb-taps-3-2.csv')  
# remove csvs
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]  
#%% load stimuli zero crossings from .npy files 

tap_optimizer = 'none' # 'random' - choose random tap, 'nearest' - choose optimal tap 
numbeats = 19

stimdirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']

stimtimbredir = './stim-no-timbre-5/'
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

    stimtimbredir =  'stim-' + tstr + '-5'

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

def parseString(st):

    _ = st.replace('[','').replace(']','')
    _ = _.split(' ')
    _ = [x for x in _ if x]
    _ = [float(x) for x in _]
        
    return np.array(_)

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
fig_agg.savefig('./plots/pc-swarm-plots.png', dpi=150)


fi.close()

#%%

















