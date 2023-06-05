#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:48:08 2023

@author: nolanlem
"""

from utils import * # utils for libraries and function defs
# set the root dir 
#os.chdir('./data/Experiment-2/scripts/')


#%% some helper string parsing functions added outside of utils
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


#%% load in main tap response datafile in pd df
df = pd.read_csv('./df-bb-taps-6-2.csv')  
# remove 2 subjects 
df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False] 

#%%##########################################################################################
####################### HISTORGRAM TAP ASYNCHRONIES ##################
##########################################################################################

numbeats = 19 # number of beats to inspect
cps = ['strong', 'medium', 'weak', 'none']
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)]


df = pd.read_csv(os.path.join('all-stats-4-7-nt-new.csv'))
df = df[df['subject'] != 'PARTICIPANT_kura-B2_2020-09-16_11h23.41.792.csv'] # remove subject
df['subject cat'] = df['subject cat'].replace({'loyalist':'regular', 'noneconvert':'hybrid', 'convert':'fast'})

df.reset_index()

fig_i, ax_i = plt.subplots(nrows=4, ncols=3, figsize=(20,8), sharex=True, sharey=True)

for k, ts in enumerate(['nt']):
    mxitipools, sxitipools = {}, {}
    for i, subcat in enumerate(['regular', 'hybrid', 'fast']):
        for n, cp in enumerate(cps):
            mxs, sxs = [], []
            mxitipools[cp], sxitipools[cp] = [], []
            for person in sorted(list(set(df[df['subject cat'] == subcat]['subject']))): 
                mxtmp = df[(df['coupling'] == cp) & (df['subject'] == person)]['subjects norm itis']
                for arr in mxtmp:
                    mxitipools[cp].extend(str_to_array(arr)[:numbeats])

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

plt.legend(title='coupling', bbox_to_anchor=(1.05, 1), loc='upper left')

#fig_i.suptitle(f'asynchronies per subject category and coupling condition')
for ax_ in ax_i[-1,:].flat:
    ax_.set_xlabel('normalized ITI')

#%%####################################################
############ K MEANS CLUSTERING ########################
############################################################

# function to perform K-means cluster on tap trial data 
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


    #### SAVE FIG#####
    #plt.savefig(os.path.join(rootdir, 'k-means-plots','new', f'K-mean-{nclusters}-clusters.eps'))
    #plt.savefig(os.path.join('./plots', f'K-mean-{nclusters}-clusters.png'),dpi=120)
    
    #finalsdir = '/Users/nolanlem/Documents/kura/swarmgen/nt-paper/images/in-paper/'
    #plt.savefig(os.path.join(finalsdir, f'K-mean-{nclusters}-clusters.eps'))
    #plt.savefig(os.path.join(finalsdir,f'K-mean-{nclusters}-clusters.tif'),dpi=120)

    return points, y_km

#%% ############################################################
######## Perform K-means Clustering  #########################
##################################################################
# Group tap ITI data into clusters as a function of the stimulus phase coherence 

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
    print(f'working on cluster {i}')
    for person in sortedkgroups[i]['subject'].values:
        idxs = sortedkgroups[i][sortedkgroups[i]['subject'] == person].index
        for idx in idxs:
            df.loc[idx, 'k group'] = i 
            

# save fig? uncomment
#plt.savefig('./plots/k-means-50.png', dpi=150)

#%%
###########################################################################
##############TIME COURSE Inter Tap Interval Analysis #####################
###########################################################################
# calculate the mean and SD of the nITIs over 7 tap sections each 3 beats long

# read in curr df

df = pd.read_csv('./all-stats-6-2-nt-new.csv');

df['subject cat'] = df['subject cat'].replace({'loyalist':'regular', 'noneconvert':'hybrid', 'convert':'fast'})


#%%

#############################################################################
############# ITI PER BEAT SEGMENT AND SUBJECT CATEGORY ######################
##############################################################################

cps = ['strong', 'medium', 'weak', 'none'] # coupling conditions
numbeats = 19 # number of beats to inspect
lenbeatsection = 3  # aggregate every 3rd beat to create section  
beatsections =  [(i, i+lenbeatsection) for i in range(0, numbeats, lenbeatsection)] # beat sections


# initialize dict of standard devs (sx) curves 
sxcurves = {}
for s_ in ['nt', 't']: # 't' is for timbre data .. not analyzing rn
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
            
            # get between subject SEM error (MX)
            mxs = np.array(mxs) 
            meanmxs = np.nanmean(mxs, axis=0)
            mxerror = np.nanstd(mxs, axis=0)
            # get between subject SEM error (SD)
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
# fig.savefig(os.path.join('./plots/', 'time-course-nITI.eps'), dpi=150,bbox_inches='tight')
# fig.savefig(os.path.join('./plots/', 'time-course-nITI.png'), dpi=150,bbox_inches='tight')


### save log plots
#fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'{ts}-{lenbeatsection}-beat-seg-log.eps'), dpi=120)
#fig.savefig(os.path.join(rootdir,'iti-plots','rev', f'loyal tap categories-beat-seg-log.png'), dpi=120)

#%%
################################################################################
#################### PHASE COHERENCE ANALYSIS ##################################
################################################################################
# perform phase coherence analysis on the aggregate tap data, calculate the 
# |R| and psi average angle. Plot via subgroup 

df = pd.read_csv('./df-bb-taps-6-2.csv')  
# remove csvs
#df = df[df["subject"].str.contains("cc_kura-B1_2020-08-09_00h01.21.424.csv") == False]
#df = df[df["subject"].str.contains("bb_kura-A2_2020-08-08_23h32.56.541.csv") == False]  
#%% l
#################################################################################
################ Koad Stimuli phase parameters from .npy files ################
#################################################################################

tap_optimizer = 'none' # 'random' - choose random tap, 'nearest' - choose optimal tap 
numbeats = 19

stimdirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']

stimtimbredir = './stim-no-timbre-5/'
unique_stimuli = sorted(list(set(df['stim'])))

nbatch = set(list(df[(df.coupling == 'none') & (df.timbre == 'n')]['stim']))
wbatch = set(list(df[(df.coupling == 'weak') & (df.timbre == 'n')]['stim']))
mbatch = set(list(df[(df.coupling == 'medium') & (df.timbre == 'n')]['stim']))
sbatch = set(list(df[(df.coupling == 'strong') & (df.timbre == 'n')]['stim']))

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
            #print(bname) 
            bname = os.path.basename(bname)
            trigsdir = os.path.join(stimtimbredir, 'stimuli_' + bname[-5], 'trigs', bname[:-4] + '.npy')
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
##################################################################
########### Generate Swarm Rose/Polar Plots ######################
##################################################################

# subset by subject cateogory (regular, hybrid, fast)
ntreg = list(set(df[(df['subject cat'] == 'regular') & (df.timbre == 'n')]['subject']))
nthyb = list(set(df[(df['subject cat'] == 'hybrid') & (df.timbre == 'n')]['subject']))
ntfast = list(set(df[(df['subject cat'] == 'fast') & (df.timbre == 'n')]['subject']))

# name of csv file to generate for stats in R, later.... 
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
            
            
            # perform circular stats test for directionality (Rayleigh test)
            p_subj = rayleightest(aggtaps%(2*np.pi))
            p_stim = rayleightest(aggzcs%(2*np.pi))
            

            # uncomment if writing all tap onsets to csv file
            # for tp in aggtaps:
            #     writer.writerow(['tap', scatstr, tnt, cp, tp])
            # for tp in aggzcs:
            #     writer.writerow(['onset', scatstr, tnt, cp, tp])
            
            # uncomment if writing tap/onset R,psi to csv file 
            writer.writerow(['onset', scatstr, tnt, cp, rm, psim])
            
            # print p value for circ. directionality per stim/tap
            print(f'{tnt} {cp} p_stim/p_subj: {p_stim}/{p_subj}' )
            
            rs, psis = getCircMeanVector(aggtaps) # get the Phase Coherence Magn.     
            writer.writerow(['tap', scatstr, tnt, cp, rs, psis])

            noise = 0.3*np.random.random(size=len(aggtaps)) # add a little noise for visual indication
            displace = 0.8
            color = 'darkblue' # subject tap trial color
            c = cm.Blues(np.linspace(0, 1, len(aggtaps)))
            tnt_str = 'no timbre'

            if k == 1:
                displace = 0.8-0.3 # displacement - noise
                color = 'darkred'
                c = cm.Reds(np.linspace(0,1,len(aggtaps)))
                tnt_str = 'timbre'
                
            ax_agg[i,j].scatter(aggtaps, displace - noise, s=20, alpha=0.05, color=c, marker='.', edgecolors='none', linewidth=0.5)
            ax_agg[i,j].arrow(0, 0.0, psis, rs, color=color, linewidth=1,  zorder=2, label=f'{tnt_str}') 
            ax_agg[0,j].set_title(f'{cp}') 
        
            
        ax_agg[i,0].set_ylabel(f'{scatstr} ') 
        i+=1 
    k+=1
plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')

plt.tight_layout()
#fig_agg.savefig('./plots/pc-swarm-plots.png', dpi=150)


fi.close()

#%%

















