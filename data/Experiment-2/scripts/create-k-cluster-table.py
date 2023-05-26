#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:03:16 2021

@author: nolanlem
"""

import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
import csv 


#%%

os.chdir('/Users/nolanlem/Documents/kura/swarmgen/mturk-csv/usable-batch-12-7/subject-raster-plots/comparisons/')
#%%
df = pd.read_csv('nt-t-subject-categories.csv')

# remove following dat files
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]

# subset data by sub cat
dfnt = df[df['timbre'] == 'nt']
dfnt_l = dfnt[dfnt['category'] == 'loyalist'] 
dfnt_nc = dfnt[dfnt['category'] == 'noneconvert']
dfnt_c = dfnt[dfnt['category'] == 'convert']


#%%
subjectgrp = 'noneconverters'

allsubjects = list(set(dfnt['subject'].values))
loyalists = list(set(dfnt_l['subject'].values))
converters = list(set(dfnt_c['subject'].values))
noneconverters = list(set(dfnt_nc['subject'].values))

if subjectgrp == 'loyalists':
    batch = loyalists
if subjectgrp == 'converters':
    batch = converters
if subjectgrp == 'noneconverters':
    batch = noneconverters


coupling = ['strong', 'medium','weak', 'none']

favg = open('./k-cluster-tables/k-cluster-table-' + subjectgrp + '-avg.csv', 'w')
writer = csv.writer(favg)
writer.writerow(['coupling', '1+2', '3','4','5'])

fstd = open('./k-cluster-tables/k-cluster-table-' + subjectgrp + '-SD.csv', 'w')
writer1 = csv.writer(fstd)
writer1.writerow(['coupling', '1+2', '3','4','5'])

avgs1 = {'strong': [], 'medium': [], 'weak': [], 'none': []}
avgs2 = {'strong': [], 'medium': [], 'weak': [], 'none': []}
avgs3 = {'strong': [], 'medium': [], 'weak': [], 'none': []}
avgs4 = {'strong': [], 'medium': [], 'weak': [], 'none': []}
avgs5 = {'strong': [], 'medium': [], 'weak': [], 'none': []}

## for loyalists, converters, noneconverts
for cp in coupling:
    for subj in batch:
        for grp in ['0']:
            grp_pct = dfnt[(dfnt['subject'] == subj) & (dfnt['coupling'] == cp)][grp]/10. 
            avgs1[cp].append(float(grp_pct))
        for grp in ['1']:
            grp_pct = dfnt[(dfnt['subject'] == subj) & (dfnt['coupling'] == cp)][grp]/10. 
            avgs2[cp].append(float(grp_pct))
        for grp in ['2']:
            grp_pct = dfnt[(dfnt['subject'] == subj) & (dfnt['coupling'] == cp)][grp]/10. 
            avgs3[cp].append(float(grp_pct))
        for grp in ['3']:
            grp_pct = dfnt[(dfnt['subject'] == subj) & (dfnt['coupling'] == cp)][grp]/10. 
            avgs4[cp].append(float(grp_pct))
        for grp in ['4']:
            grp_pct = dfnt[(dfnt['subject'] == subj) & (dfnt['coupling'] == cp)][grp]/10. 
            avgs5[cp].append(float(grp_pct)) 

avgs_ = [avgs1, avgs2, avgs3, avgs4, avgs5]            

for cp in coupling:
    grpavgs, grpstds = [], []
    for i, avg_ in enumerate(avgs_):
        avg = np.mean(avg_[cp])
        std = np.std(avg_[cp])
        grpavgs.append(avg)
        grpstds.append(std)
        
    merged_avgs = [cp, grpavgs[0]+grpavgs[1], grpavgs[2], grpavgs[3], grpavgs[4]]
    
    merged_stds = [cp, grpstds[0]+grpstds[1], grpstds[2], grpstds[3], grpstds[4]]

    writer.writerow(merged_avgs)
    writer1.writerow(merged_stds)
        
    print('')
            
favg.close()
fstd.close()