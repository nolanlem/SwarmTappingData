
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:16:58 2019
 This program generates different version of coupled oscillator audio output where samples of a woodblock metronome sound are triggered at each oscillator's zero crossing. This was used to generate auditory stimuli for
 the study 'Individual differences of limitation to extract beat from Kuramoto coupled oscillators: transition from beat-based tapping to frequent tapping with weaker coupling'.
 
The generative model uses a Kuramoto Model to synchronize 40 oscillators at varying degrees of coupling. 
The phase coherence (R) is fedback in the generative loop to stabilize the synchrony of the system depending
 on the coupling coefficient. 

@author: nolanlem
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import librosa
import soundfile as sf
import sys
import string

from scipy.signal import hilbert, chirp
import os 
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import datetime
import itertools
import glob 

import config_kuramoto as config

sns.set()
pi = np.pi

#%% #### UTIL FUNCTIONS

def freq2rad(freqs, sr):
    freqs = freqs*2*np.pi/sr
    return freqs

def frame2samp(frames, hop):
    # linear interpolation
    samples = np.zeros((frames.shape[0], frames.shape[1]*hop))
    for i in range(frames.shape[0]):
        for j in range(1,frames.shape[1]):
            samples[i,(j-1)*hop:(j*hop)] = np.linspace(frames[i,j-1], frames[i,j], hop)
    return samples

# create gaussian for intrinsic freq
def gaussian(numoscs, filename, mu=0.00, sigma=0.1, plotFlag = False):
    #mu, sigma = 0.00, 0.1 # mean and std deviation
    dist = np.random.normal(mu, sigma, numoscs);
    # sanity check the mean and variance
    
    if plotFlag == True:
        abs(mu - np.mean(dist)) < 0.01
        abs(sigma - np.std(dist, ddof=1)) < 0.01
        # generate histogram
        count, bins, ignored = plt.hist(dist, 30, density=True)
    return dist

# beat binning 
def binTapsFromBeatWindow(taps):
    binnedtaps = []
    for i, tap in enumerate(taps):
        try:
            binnedtaps.append(taps[i][0]) # take first tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    return binnedtaps
    
# beat binning 
def formFixedBeatBins(wf, thesnd, limitpeaks=False):
    strsnd= os.path.basename(thesnd).split('.')[0]
    beat_bins = 0
    amp_peaks, _ = find_peaks(wf, height=0.14, distance=sr/3.0) # get amplitude envelope and return peaks
    #print '\t num peaks %r = %r'%(strsnd, len(amp_peaks))
    avg_int_bb = librosa.samples_to_time(np.average(np.diff(amp_peaks)))    
    
    # get the ideal period from the sound file path
    idealperiod = 60./float(os.path.basename(thesnd).split('.')[0].split('_')[1])
    fixed_bb = [avg_int_bb*i for i in range(len(amp_peaks))]

    # none of the audio should have more than 15 'beats' therefore if find_peaks
    # returns too many peaks, use the 'idealperiod' to create the fixed beat window array
    if limitpeaks==True:
        if len(amp_peaks) > 15:
            print('too many amp peaks')
            fixed_bb = [idealperiod*i for i in range(14)]

    avg_bpm = 60./avg_int_bb # save the avg bpm depending on avg period
    beat_bins = librosa.samples_to_time(amp_peaks) # convert to samples

    # shift over fixed beat window depending on if its > or < first amplitude env peak 
    if fixed_bb[0] < beat_bins[0]:
        fixed_bb += (beat_bins[0] - fixed_bb[0])
    if fixed_bb[0] > beat_bins[0]:
        fixed_bb -= (fixed_bb[0] - beat_bins[0])
    # shift over half window (makes the "GT beat" at 180 deg in phase coherence plots )
    for i in range(1, len(fixed_bb)):
        fixed_bb[i-1] = fixed_bb[i-1] + (fixed_bb[i] - fixed_bb[i-1])/2 
    fixed_bb[-1] = fixed_bb[-1] + avg_int_bb/2. # last in array
    
    if fixed_bb[0] >= avg_int_bb:
        fixed_bb = np.insert(fixed_bb, 0, fixed_bb[0] - avg_int_bb)
    return fixed_bb, avg_bpm

#### helper functions

# function to generate the complex order params (R, psi)
def accumulateR(ph, numoscs):
#    function to  complex order params
    # r = 0
    r = np.sum(np.exp(ph*1j))
    r = r/numoscs # divide by number of active oscillators
    rang = np.angle(r) # get avg phase angle, rang
    rmag = np.abs(r) # get phase coherence, rmag
    return [rang, rmag] # return complex order params

# function to update phases in gen model 
def updatePhases(ph, freq, pang, orderparams, kn):
    # function to update phases of each oscillator in ensemble
    rang, rmag = orderparams
    pang = kn*rmag*np.sin(rang-ph)
    ph = ph + freq + pang
    ph = ph%(2*np.pi)
    # return the phases
    return ph

# get longest list in list of lists python 
def longest(l):
    if(not isinstance(l, list)): return(0)
    return(max([len(l),] + [len(subl) for subl in l if isinstance(subl, list)] +
        [longest(subl) for subl in l]))

# make dir if it doesn't exist 
def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname) 
        
# beat binning functions
def binBeats(taps, beat_bins): 
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    for i, tap in enumerate(taps):
        try:
            binnedtaps.append(taps[i][-1]) # take last tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    return binnedtaps
      
    
# function to determine beat bins and plot with waveform from psi 
    
def getBeatBinsFromPsi(filenames, savePlots=False):
    print('--------now getting beat bins from psi----------')
    # cond = the sound in this 'weak_94_1' format
    fig_, ax = plt.subplots(len(filenames),1, figsize=(20,20),sharex=True,sharey=True)
    
    beatbins = {}
    sndrefs = []
    
    for i, filen in enumerate(filenames):
        cond = os.path.basename(filen).split('.')[0]
        sndrefs.append(cond)
        print('analyzing %r/%r: %r' %(i,len(filenames),cond))
        
        filepath = dirforstim + 'stimuli_'+ cond[-1] + "/" 
        zcs = np.load(filepath + "phases/" + cond + ".npy" , allow_pickle=True)
        psi = np.loadtxt(filepath + "phases/ang/" + cond + ".txt", delimiter=",") 
        y, sr_ = librosa.load(filepath + cond + ".wav", sr=audio_sr)
    
        cutin  = 0
        cutout  = y.size # whol audio or take first 15 seconds, int(15*audio_sr)
        totals = np.sum(zcs, axis=0)/zcs.shape[0] # /N
        totals = totals[cutin:cutout]
        #plt.plot(totals)
        y = y[cutin:cutout]
        psi = psi[cutin:cutout] # do this if COP at audio sr already 
    
        angular_vel = np.unwrap(psi)    # unwrap phase
        angular_vel = np.sin(angular_vel) 
    
        # plot    
        #plt.figure(figsize=(20,5))
        ax[i].plot(angular_vel, color='green')
        amp_peaks, _ = find_peaks(angular_vel, height=0.75, distance=sr/2.0) # get amplitude envelope and return peaks
        mean_diff = np.mean(np.diff(amp_peaks)) # get average time between windows
        shift = int(mean_diff/6) #
        amp_peaks += shift # shift beat windows over to center beat
        ax[i].vlines(amp_peaks,-1,1, color='red') # plot the beat bins 
        beatbins[cond] = amp_peaks
        ax[i].plot(y, linewidth=0.5)
        
        mean_bpm = 60./(mean_diff/audio_sr) 
        mean_bpm = str(round(mean_bpm, 1))
        thetitle = cond + ' ' + 'avg bpm: ' + mean_bpm
        ax[i].set_title(thetitle)
        
    
    timetag = str(datetime.datetime.now())    
    plt.savefig(plotdir + 'beat-windows/' + timetag + ".png", dpi=150)
    return beatbins, sndrefs 

#%%#############################################################################
################## GENERATIVE KURAMOTO MODEL CONFIGURATION #####################
#################################################################################

#####NB: for each instantiation  #################
#numbers = [1,2,3,4] # whow many versions number to make, NB: have to change and rerun block below 

### INITIALIZE MODEL PARAMETERS #####
numbers = [1] # just 2 version 
syncs = ['none', 'weak', 'medium', 'strong'] # sync conds
#bpms = np.geomspace(72, 120, num=5) # bpms from 60 - 120 bpm on log scale
bpms = np.geomspace(72, 120, num=1) # bpms from 60 - 120 bpm on log scale
bpm_str = [str(int(elem)) for elem in bpms]
  
kn_coeffs = [0.0, 0.12, 0.15, 0.3] # initial coupling coefficients, kn
R_target = [0.0, 0.4, 0.6, 0.99999] # phase coherence target, for feedback
kn_scalar = 0.01 # increment for each 
adaptive_window = 10 # number of frames at gen model sr, 10 frames=0.5 sec for R feedback 
   
sr = 20           # generative model samp rate
audio_sr = 22050. # audio sr
numoscs = 40      # N, total num of oscillators in group
seconds = 45      # total seconds - sample_cutoff time
cutin_sec = 10     # how long to 'wait' to start 'recording' 
sample_cutoff = int(audio_sr*cutin_sec) # wait five seconds for system to settle into regime
    

dirforstim = "./gen-stim-output/" # dir to store stim params and audio output
if os.path.exists(dirforstim) == False:
    os.path.mkdir(dirforstim)


for number in numbers:
    bigtime1 = time.time()
    
    #form strings for beat bin plotting
    conds = []
    
    print('with kn initial: ', str(kn_coeffs))
    for bpm in bpms:
        for sync, kn, r_tar in zip(syncs, kn_coeffs, R_target):
            print(str(sync) + "_" + str(int(bpm)) + "_" + str(number) + "-" + str(r_tar))
            conds.append(str(sync) + "_" + str(int(bpm)) + "_" + str(number))
            
    plotdir = os.path.join(dirforstim, "stimuli_" + str(number), 'plots')
    
    # create dirs for audio and data 
    if os.path.exists(dirforstim) == False:
        print('making dirs for', dirforstim)
        os.mkdir(dirforstim)
    
            
    audio_dir = os.path.join(dirforstim, "stimuli_" + str(number))
    phases_dir =  os.path.join(audio_dir,'phases')
    pc_dir = os.path.join(phases_dir, 'pc')
    psi_dir = os.path.join(phases_dir,'ang')
    centerbpm_dir = os.path.join(phases_dir, 'center-bpm')
    trigs_dir = os.path.join(audio_dir,'trigs')
    
    plot_dir = os.path.join(dirforstim, "stimuli_" + str(number), "plots")
    
    # if this version hasn't been created, make a new dir to store audio,plots,etc. 
    # print out all the stimuli strings that will be generated    
    if os.path.exists(dirforstim + "stimuli_" + str(number)) == False:
        print('making dirs for', dirforstim + "stimuli_" + str(number))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number)))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "mp3s"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "phases"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number),"trigs"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "phases", "ang"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "phases", "pc"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "phases", "init_freqs"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number),  "phases", "init_phases"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number),"phases", "center-bpm"))
        os.mkdir(os.path.join(dirforstim, "stimuli_" + str(number), "phases", "beat-windows"))
        os.mkdir(os.path.join(plot_dir, "beat-windows"))

       
    # for plots 
    if os.path.exists(plotdir) == False:
        os.mkdir(plotdir)
        os.mkdir(os.path.join(plotdir, 'beat-windows'))
        os.mkdir(os.path.join(plotdir,'cop'))
  
#   #######################################################################
    ################ MAIN GENERATION LOOP ####################
    ########################################################################
    print('\n \n ######## MAIN GENERATIVE LOOP ############')
    time1 = time.time()
    
    cnt = 1
    totalcnt = len(conds)
    optimal_kns = {}
    center_period = {}
    
    for bpm in bpms:
        for sync_, kn_, rtar in zip(syncs, kn_coeffs, R_target):
            innertime1 = time.time()
            print('working on %r/%r, \n bpm: %r \t coupling: %r \t init_kn: %r \t target_R: %r' %(cnt, totalcnt, bpm, sync_, kn_, rtar))
                       
            tempo = str(int(bpm))
            freqmult = bpm/60. # to be applied to frequencies to speed up or slow down ensemble for each tempo cond
            kn = kn_ # kn_coeffs
            # freq of oscillators
            #f_mean = bpm/60.    # 1.6
            f_mean = 1.
            #f_dev = f_mean*0.2667
            #f_dev = 1/(2*f_mean)
            f_dev = 0.4 # init frequency variance 
            freq_hz = gaussian(numoscs, '', f_mean, f_dev)
            freq_hz = freq_hz*freqmult
            
            init_freqs = [round(elem,3) for elem in freq_hz]
            freq_bpm = 60./(1./np.mean(freq_hz)) # init freq in bpm 
            freq = freq2rad(freq_hz, sr=sr) # convert to radians
                      
            # initial phase stuff
            p = 2*np.pi*np.random.rand(numoscs) # random initial phases
            init_phases = np.copy(p)
            init_phases = [round(elem,2) for elem in init_phases] # for saving to txt
            
            pang = np.zeros(numoscs)
            mean = config.config['mean']
            std = config.config['std']
    
            phases, samples, phasecoherence, avgphase  = [],[],[],[] # empty lists to write to txt
        
            avgdur = 0.0
            totaliters = sr*seconds
            totaldurationsec = totaliters/sr
            # sanity check params
            print('PARAMETERS: \n numoscs: %r kn: %r mean: %r std: %r' %(numoscs, kn, np.round(mean,3), np.round(std,3)))
    
            # empty audio buffer
            thesample = './samples/woodblock_mono.wav'
            totalsamples = np.int(totaldurationsec*audio_sr)
            sample, _ = librosa.load(thesample, sr = audio_sr)
            sample = (1./20)*sample 
            
            sample_len = np.shape(sample)[0]
            audiobuffer = np.zeros(totalsamples+sample.shape[0], dtype='float32')
            triggerbuffer = np.zeros((numoscs, totalsamples)) # to be saved for data analysis later
            
            trigsbuffer = [[] for i in range(numoscs)]
            ########################################################################
            ##################### main generative loop ###########################################
            
            R_mean_arr = [] # list to keep track of phase coherence mean over window  (for R feedback)
            for x in range(totaliters):
                # accumulate phase coherence, R
                complexorderparams = accumulateR(p, numoscs)
                tempR = complexorderparams[1]
                R_mean_arr.append(tempR) # append to list 
                
                # feedback from pc: adapt kn to target R value over 1 second intervals to stabilize synchrony at output
                if (x % adaptive_window == 0):
                    R_mean_sec = np.mean(R_mean_arr)
                    if (R_mean_sec < rtar):
                        kn += kn_scalar 
                    if (R_mean_sec > rtar):
                        kn -= kn_scalar
                    R_mean_arr = [] 
                
                # don't adapt kn for strong and none conds
                if (sync_ == 'strong'):
                    kn = 1.2
                if (sync_ == 'none'):
                    kn = 0.0
    
                phasecoherence.append(complexorderparams[1])
                avgphase.append(complexorderparams[0]) 
                
                #def updatePhases(ph, freq, pang, orderparams, kn):
                p_ = np.copy(p) # p[x-1]
                p = updatePhases(p, freq, pang, complexorderparams, kn) #p[x]
    
                for i in range(len(p)):
                    if p[i] < p_[i]: # check for zero crossing
                        #print '%r crossing' %(i)
                        pos = np.int(audio_sr*(x/float(sr)))  # this is the upsampling step
                        audiobuffer[pos:(pos+sample_len)] = audiobuffer[pos:(pos+sample_len)] + sample
    
                        triggerbuffer[i, pos:pos+1] = 1 # load up phases of each osc for 'ground truth' phase coherence (later)
                        trigsbuffer[i].append(pos)
                        #triggerbuffer[i].append(pos) # or just save sample numbers
                    
                # this is saving the comp ord parameter at audio sr ...
                for i in range(int(totalsamples/totaliters)):
                    phasecoherence.append(complexorderparams[1])
                    avgphase.append(complexorderparams[0])                
      
                       
                p_ = p.tolist()
                phases.append(p_)
                samples.append(np.cos(p_))
                #kn += kn_to_span/totaliters
    
            phasecoherence = np.array(phasecoherence[int(sample_cutoff):])
            avgphase = np.array(avgphase[int(sample_cutoff):])
    
            # add 0.5 sec fade in, fade out 
            fadein_env = np.linspace(0, 1, int(audio_sr/2))
            fadeout_env = np.linspace(1, 0, int(audio_sr/2))
            ones = np.ones(len(audiobuffer) - len(fadein_env) - len(fadeout_env))
            fades_env = np.append(fadein_env, ones)
            fades_env = np.append(fades_env,fadeout_env)
            # add env to sound 
            audiobuffer = audiobuffer*fades_env
            audiobuffer_stereo = np.array([audiobuffer, audiobuffer])
            audiobuffer_stereo = np.asfortranarray(audiobuffer_stereo[:, int(sample_cutoff):]) # take out first 5 seconds       
            sync_tempo = sync_ + "_" + tempo + "_" + str(number)
            optimal_kns[sync_tempo] = kn
            center_period[sync_tempo] = 1./(freq_bpm/60.)
    
            ################ write audio and params to file ###################
            # save output audio
            sf.write(os.path.join(dirforstim, "stimuli_" + str(number), sync_tempo + ".wav"), audiobuffer_stereo.T, samplerate=22050) 
            #librosa.output.write_wav(home + "stimuli_" + str(number) + "/" + sync_tempo + ".wav", audiobuffer_stereo, sr=int(audio_sr), norm=False)
            print('writing file: ', dirforstim + "stimuli_" +  str(number) + "/" + sync_tempo + ".wav")           
            triggerbuffer = triggerbuffer[:, int(sample_cutoff):]             
            # save raw phase data per oscillator per time step 
            np.save(dirforstim + "stimuli_" +  str(number) + "/phases/" + sync_tempo + ".npy", triggerbuffer)                    
            # save trigs, zero crossings
            np.save(dirforstim + "stimuli_" +  str(number) + "/trigs/" + sync_tempo + ".npy", trigsbuffer)                    
            # save the phase coherence
            np.savetxt(dirforstim + "stimuli_" + str(number) + "/phases/pc/" + sync_tempo + ".txt", phasecoherence, delimiter=',')            
            # save the avg angle
            np.savetxt(dirforstim + "stimuli_" +  str(number) + "/phases/ang/" + sync_tempo + ".txt", avgphase, delimiter=',')
            
            # save center freq, init phases, init freqs        
            np.savetxt(dirforstim + "stimuli_" +  str(number) + "/phases/center-bpm/" + sync_tempo + ".txt", np.array([round(freq_bpm,2)]), delimiter=',')        
            np.savetxt(dirforstim + "stimuli_" +  str(number) + "/phases/init_phases/" + sync_tempo + ".txt", init_phases, delimiter=',')
            np.savetxt(dirforstim + "stimuli_" +  str(number) + "/phases/init_freqs/" + sync_tempo + ".txt", init_freqs, delimiter=',')
            
            print('final kn value was:', np.round(kn,3))        
            print('writing file to: ' + dirforstim + "stimuli_" + str(number) + "/" + sync_tempo + ".wav" )
            print("\n \n")
            innertime2 = time.time()
            print('---> took %r seconds \n' %(np.round(innertime2-innertime1),3))
            cnt += 1 
    
    time2 = time.time()
    
    print(f'doneskies, took {np.round((time2-time1)/60,3)} minutes')
    
#%%
########################################################################    
###################  BEAT WINDOW Generation and Analysis ################   
########################################################################    
#########################################################################    
   
 

############### GET BEAT BINS FOR A BATCH OF THE STIMULI
# get beat bins (bbs) and sndrefs (sndfile names)
# allbatch = sorted(conds) # get beat bins for all stimuli 
# # by coupling condition
# strongbatch = [elem for elem in conds if elem.startswith('strong')]
# medbatch = [elem for elem in conds if elem.startswith('med')]
# weakbatch = [elem for elem in conds if elem.startswith('weak')]
# nonebatch = [elem for elem in conds if elem.startswith('none')]

# allbatch = nonebatch + weakbatch + medbatch+ strongbatch
# all_beat_bins, sndrefs = getBeatBinsFromPsi(allbatch, savePlots=False)
thebatch = conds
all_beat_bins, sndrefs = getBeatBinsFromPsi(thebatch, savePlots=False)

#################filter out beat bins that are too close together from peak picking 

fig, ax = plt.subplots(len(thebatch), 1, figsize=(10,10))
beatbins_filtered = {}

for j,snd in enumerate(thebatch):
    print('working on ', snd)
    # v1 
    tempo_cond = snd.split(sep="_")[1]
    tempo_cond_ = float(tempo_cond)
    # v2 by center bpm  
    tempo_cond = np.loadtxt(audio_dir + "/phases/center-bpm/" + snd + ".txt", delimiter=',')        
    tempo_cond = audio_sr*1./(tempo_cond/60.)
    tempo_cond = 0.67*tempo_cond # can't be smaller than half the average bpm
    #print 'tempo cond_: %r \t tempo cond: %r'%(tempo_cond_, tempo_cond)
    
    min_threshold_beatbin_size_ = int(0.67*audio_sr*60./tempo_cond_) # constrain bb to be < 0.5 of bpm center (in samples)
    min_threshold_beatbin_size = int(tempo_cond)
    #print 'tempo cond_: %r \t tempo cond: %r'%(min_threshold_beatbin_size_, min_threshold_beatbin_size)

    beatbins = all_beat_bins[snd]
    beatbins_diffs = np.diff(beatbins)
    idx_to_delete = []
    for i, diff in enumerate(beatbins_diffs):
        if diff < (min_threshold_beatbin_size_):
            idx_to_delete.append(i)
    
    beatbins_filtered[snd] = np.delete(beatbins, idx_to_delete)
    
    y, sr_ = librosa.load(os.path.join(audio_dir, snd + ".wav"), sr=int(audio_sr))
    ax[j].plot(y, linewidth=0.5)
    ax[j].vlines(beatbins_filtered[snd], -1, 1, color='red')            
            
     
    

#######  use the beat_window locations to crop out first 23 beats of audio stim

beat_25_sample_cutoff = {} # now just 23 beats...
beat_bins_cropped = {}

# for weakbatch, or allbatch 
for snd in thebatch:
    thesnd = os.path.basename(snd).split('.')[0]
    
    beat_bins_cropped[snd] = beatbins_filtered[snd][:20]
    np.savetxt(dirforstim + "stimuli_" +  str(number) + "/phases/beat-windows/" + thesnd + ".txt", beat_bins_cropped[snd], delimiter=',')
    
    beat_25_sample_cutoff[snd] = beat_bins_cropped[snd][-1]
    #print beat_23_sample_cutoff[snd]
    stimpath = audio_dir + "/" + snd + ".wav"
    #print stimpath
    y, sr_ = librosa.load(stimpath)  
    y = y[:beat_25_sample_cutoff[snd]]
    #librosa.output.write_wav(stimpath, y, sr=int(audio_sr), norm=True)
    sf.write(stimpath, y.T, samplerate=22050)
        

################# plot cropped audio waveform with first 25 beat bins 

fig, ax = plt.subplots(len(thebatch), 1, sharey=True, sharex=True,figsize=(10,10))
for i,snd in enumerate(thebatch):
    print('working on %r' %(snd))
    stimpath = dirforstim + "stimuli_" + str(number) + "/" + snd + ".wav"    
    y, sr_ = librosa.load(stimpath)
    ax[i].plot(y, linewidth=0.4) 
    ax[i].vlines(beat_bins_cropped[snd],-1,1,color='red')
    ax[i].set_xticklabels('')
    ax[i].set_yticklabels('')
    ax[i].set_title(snd)

plt.savefig(os.path.join(plotdir, 'beat-windows', 'all-stim-25-beat-windows.png'), dpi=150)

        

################### if trigger buffer is audio_sr len
############## get all the phases (GM "taps") ref by sndfile name ('vweak_60_1')
phases = {} 
sndbatch = []
cnt=1
for fi in glob.glob(os.path.join(phases_dir,'*.npy')):
    print('loading phases from %r/%r : %r'% (cnt,len(conds), os.path.basename(fi).split('.')[0]))
    phase = np.load(fi)
    thesnd = os.path.basename(fi).split('.')[0]
    phases[thesnd] = phase 
    sndbatch.append(thesnd) 
    cnt+=1



##### Function to plot Complex Order Params on Circle Map 

def plotCOP(batch):
    print('--------plotting COPs---------')
    synctemp_pool = []
    complexorderparams = {}
    k = 0
    plt.figure(figsize=(18,5))
    for snd in batch:
        print('plotting COPs for %r...'%snd)
        
        stim_pooled_taps = [] # array to hold all taps for one stimuli in loop
        beat_bins = all_beat_bins[snd] 
        #beat_bins = np.insert(beat_bins, 0,0)
        for taps in phases[snd]:
            tap_pos = np.nonzero(taps)[0] # return index where tap (1) is 
            bins = binBeats(tap_pos, beat_bins) # place taps in correct beat bin 
            binnedTaps = binTapsFromBeatWindow(bins) # take first tap in each beat window/bin
            
            binnedTaps = [x for x in binnedTaps if x < beat_bins[-1]] # remove taps outside of last beat window
            
            #tap_pos = [x for x in tap_pos if x < beat_bins[0] ]
            #binnedTaps = binnedTaps[0]
            
            lenbt = len(binnedTaps)
            lenbb = len(beat_bins)
            if lenbb <= lenbt:
                end = lenbb
            else:
                end = lenbt
            
            circletaps = [] # tap pool to add mapped values (0-2pi)
            for i in range(1, end):
                binmin = beat_bins[i-1] # left beat window 
                binmax = beat_bins[i]   # right beat window
                bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) # interpolate zero crossing wrt beat bin to give value 0-2pi
                #breakpoint()
                circletaps.append(float(bininterp(binnedTaps[i-1]))) # append binned taps to circle map list
           
            circletaps = np.array(circletaps) # convert to np array
            flattaps = circletaps[~np.isnan(circletaps)] # remove NaNs
            stim_pooled_taps.append(flattaps) # add to stim_pooled_taps for each sync condition
    
        stim_pooled_taps = list(itertools.chain(*stim_pooled_taps))    # flatten 2D list of pooled taps for the snd stimulus
        R = np.abs(np.nanmean(np.exp(1j*np.array(stim_pooled_taps)))) # get phase coherence, R
        psi = np.angle(np.nanmean(np.exp(1j*np.array(stim_pooled_taps)))) # get avg angle, psi
        synctemp_pool.append(stim_pooled_taps)
        
        ## RAYLEIGH test for non-uniformity 
        # should try Kuiper's test.....
        #pval = rayleightest(np.array(stim_pooled_taps))
        #print 'Rayleigh's test: pval = ', np.round(pval,5)
        
        complexorderparams[snd] = {}
        complexorderparams[snd]['R'] = R
        complexorderparams[snd]['psi'] = psi
    
        ax = plt.subplot(1, len(batch) , k+1, projection='polar')            
        randnoise = 0.7+np.random.rand(len(stim_pooled_taps))*0.3 # to distribute dots visually so they pile
        colors = np.linspace(0,100,len(stim_pooled_taps))
        ax.scatter(stim_pooled_taps, randnoise, s=5, alpha=0.75, c=colors)
        #ax.bar(stim_pooled_taps, 1,width=2*np.pi/len(stim_pooled_taps), )
        #ax.set_axisbelow(True)
        ax.grid(linewidth=0.1, alpha=1.0)    
        ax.arrow(0, 0.0, psi, R, color='black')
        ax.set_ylim(0,1.0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        textstr = 'R =' + str(round(R,2)) + '\n' + u'\u03C8=' + str(round(psi,2))
        ax.set_title(os.path.basename(snd) + '\n' + textstr, size=8, rotation=45,pad=30)
    
        k+=1                 
    
    timetag = str(datetime.datetime.now())
    try:
        plt.savefig(plotdir + 'cop/' + timetag + '.png', dpi=150) 
    except IOError:
        print('couldnt save file')
    
    return complexorderparams
  
####### REPLOT COP after update
cops = plotCOP(thebatch)
################ order stimuli by phase coherence(R) in ascending order 
from collections import OrderedDict
R_ascending = OrderedDict(sorted(cops.items(), key=lambda kv: kv[1]['R']))
R_ordered = list(R_ascending.keys()) # return just sndfile name e.g. 'strong_90_1'

Rs = []
snds_R = []
for i,snd in enumerate(R_ordered):
    the_R = R_ascending[snd]['R']
    Rs.append(the_R)
    snds_R.append(snd)
    print(snd, the_R)


############# plot COPs by phase coherence
fig, ax = plt.subplots()
numbers = np.linspace(1, len(Rs), len(Rs))
ax.plot(numbers, Rs, marker='o')
ax.set_ylabel('phase coherence (R)')
ax.set_xlabel('snd')

for i, txt in enumerate(numbers):
    ax.annotate(txt, (numbers[i], Rs[i]))

plt.savefig(plotdir + "sorted-by-R.png",dpi=150)
############### save a .csv file of the list 
import csv 
with open(plotdir + 'R-ascending-new-25.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["number", "snd","R"])
    nums = np.linspace(1,len(Rs),len(Rs), dtype=np.int)
    for i, R, snd in zip(nums, Rs, snds_R):
        writer.writerow([i, snd, str(round(R,3))])
############ now can go back and plotCOP in ascending R order
cops = plotCOP(snds_R) 

bigtime2 = time.time()
print('\n \n ------- the whole program takes: %r minutes ------' %(np.round((bigtime2-bigtime1)/60., 3)))

################# END ##############################


#%% # amp env peak finding 


from scipy.signal import hilbert 
from scipy import signal 
import glob

numsecs = 5

sndbatch =[]
for fi in glob.glob('./stim-no-timbre-5/stimuli_4/*.wav'):
    sndbatch.append(fi)

fig, ax = plt.subplots(2,2,figsize=(10,7), sharex=True)
for i, snd in enumerate(sndbatch[:2]):
    y, _ = librosa.load(snd,sr=audio_sr)
    y = y[:int(numsecs*audio_sr)]
    
    ax[i,0].plot(y, color='orange')
    
    fc = 2000  # Cut-off frequency of the filter
    w = fc / (audio_sr / 2) # Normalize the frequency
    b, a = signal.butter(6, w, 'low')
    filtered_sig = signal.filtfilt(b, a, y)
    
    ax[i,0].plot(y)
    ax[i,0].plot(filtered_sig)
    
    analytic_signal = hilbert(filtered_sig)
    
    amp_env = np.abs(analytic_signal)
    #ax[i,0].plot(amp_env)
    
    amp_peaks, _ = find_peaks(y, height=0.72, distance=sr/2.0) # get amplitude envelope and return peaks
    
    #fixed_bb, avg_bpm = formFixedBeatBins(filtered_env, thesound)
    
    #ax[i].figure()
    ax[i,1].plot(y)
    ax[i,1].vlines(amp_peaks, -1, 1, color='red', linewidth=0.5)
    ax[i,1].set_xticks(np.arange(0, len(y), 44100))
    ax[i,1].set_xticklabels([str(elem/44100) for elem in np.arange(0,len(y), 44100)])



#%%

