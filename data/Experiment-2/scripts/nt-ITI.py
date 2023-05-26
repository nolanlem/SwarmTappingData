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


os.chdir('/Users/nolanlem/Documents/kura/swarmgen')

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
    df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv'] #remove this subject
    df.reset_index()    
    
    n_per_cat = getNumberSubjectsPerCat(df)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8), sharex=True, sharey='row')

    fagg = open(os.path.join(rootdir, 'csvs', ts + '.csv'), 'w')
    writer2 = csv.writer(fagg)
    writer2.writerow(['subcat', 'coupling', 'tap section', 'sx'])
    
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
                
                writer2.writerow([subcat, cp, b, meanmxs[b]])
                
            xr = np.arange(0+n/7, len(beatsections)+n/7, 1)
            ax[0,i].errorbar(x=xr, y=meanmxs, yerr=mxerror, label=cp, capsize=4, marker='o') 
            #ax[0,i].legend()
            ax[0,i].set_title(f'{subcat} N={n_per_cat[subcat]}')
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
            
            # ax[1,i].set_yticks([0.00,0.25])
            # ax[1,i].set_yticklabels([0.00, 0.25])
            
            ax[1,i].errorbar(x=xr, y=meansxs, yerr=sxerror, label=cp, capsize=4, marker='o') 
            #ax[1,i].legend()
            #ax[1,i].set_title(f'{subcat}')
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
    fig.legend(handles, labels, loc='right')
    #fig.suptitle(f'{ts} nITI between subject groups: {len(beatsections)} segments, each with {lenbeatsection} beats')
  
f.close()
fagg.close()
f3.close()
# # save normal plots      
# fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg.eps'), dpi=120,bbox_inches='tight')
# fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg.png'), dpi=120,bbox_inches='tight')

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
def compute_gof(y, y_fit):
    ss_res = np.sum((y-y_fit)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

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
    fi = open(os.path.join(csvdir, ts + '-params-.csv'), 'w')
    writer = csv.writer(fi)
    writer.writerow(['group', 'coupling', 'A', 'B'])
    
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
            ax[s].plot(t, z[0][0]*np.exp(z[0][1]*t), label=cp, marker='x', color=sns.color_palette('tab10')[n]) 
            curveparams[ts][subcat][cp] = z[0]
            
            
            ## method 3
            # popt, pcov = curve_fit(func, t, curve, maxfev=5000)
            # ax[s].plot(t, func(t,popt[0], popt[1], popt[2]), label=cp, marker='x')
          
            ax[s].plot(t, curve, color=sns.color_palette('tab10')[n], linewidth=0.5)
            ax[s].set_xlabel('Tap Section')
            ax[s].set_title(subcat) 
            
            
            # write params to csv file 
            writer.writerow([subcat, cp, z[0][0], z[0][1]])
            
    plt.legend()
    plt.savefig(os.path.join(csvdir, ts + '-plot.png'), dpi=120)
    
fi.close()

#%%
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

plt.savefig(os.path.join(csvdir, 'Rstim_vs_params.png'), dpi=120)

            

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


#%% 
print('loyalists')
loyalists = set(df[df['subject cat'] == 'loyalist']['subject'])
for _ in loyalists:
    print("'" + _ + "',")
    
print('\nhybrids')
hybrids = set(df[df['subject cat'] == 'noneconvert']['subject'])
for _ in hybrids:
    print("'" + _ + "',")
    
print('\nfast')
hybrids = set(df[df['subject cat'] == 'convert']['subject'])
for _ in hybrids:
    print("'" + _ + "',")



