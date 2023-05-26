#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:11:44 2022

@author: nolanlem
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


rootdir = './mturk-csv/usable-batch-12-7/subject-raster-plots/nt/all/'
df = pd.read_csv(os.path.join(rootdir, 'dispersion-plots', 'all-subjects', 'subjects-disp-centroid-all.csv'))

df = pd.read_csv('/Users/nolanlem/Documents/kura/final-tapping-scripts/timbre-paper/df-bb-taps-3-2.csv')



df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]


df_all = df[(df['k group'] == 2.0) & (df.timbre == 'n')]['dispersion']
df_all = df[((df['dispersion group'] == '3d') | (df['dispersion group'] == '3s')) & (df.timbre == 'n')]['dispersion']




#%% Create plot for Supplementary Materials Dispersion Historgram 
plt.figure(figsize=(6,4))
ax = plt.gca()
ax.set_xlim([0,0.5])

ax.hist(df_all, color=sns.color_palette()[2], bins=188, linewidth=0.2)
threshold = 0.12
ax.set_ylabel('Count')
ax.set_xlabel('Tap Dispersion')

ax.vlines(threshold, 0, 25, color='red', linewidth=0.8, linestyle='dashed')

#plt.savefig('tap-disp-supplementary-materials.png', dpi=150)
plt.savefig('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/in-paper/disp-histo.png', dpi=150)



#%%
#plt.hist(df[df['k means grp'] == 3.0]['tap disp'], bins=100)
#@plt.xlim([0,0.4])

fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(6,6), sharex=True, sharey=True)

df_all = df[df['k means grp'] == 3.0]['tap disp']

df_slice1 = df[(df['k means grp'] == 3.0) & (df['tap disp'] <= threshold)]['tap disp']

df_slice2 = df[(df['k means grp'] == 3.0) & (df['tap disp'] > threshold)]['tap disp']


ax[0].hist(df_all, bins=200)
ax[0].set_title('k=3 all tap dispersion')
ax[1].hist(df_slice1, bins=20, linewidth=0.1)
ax[1].set_title(f'k=3 threshold <= {threshold}')
ax[2].hist(df_slice2, bins=200)
ax[2].set_title(f'k=3 threshold > {threshold}')

for ax_ in ax.flat:
    ax_.set_xlim([0,0.4])

#ax[3].hist()


#plt.savefig(os.path.join(rootdir, 'dispersion-plots', 'all-subjects', 'thresholded-disp.png'), dpi=150)


