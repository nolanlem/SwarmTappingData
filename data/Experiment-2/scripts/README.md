The main analysis scripts are in data-analysis.py which analysize the tap asynchronies, k-clustering of normalized inter-tap-intervals, time course analysis and phase coherence. The utils.py file contains libraries and function definitions.  

The main generative modeling scripts are contained in generate-stim.py which reproduces the stimuli used in this experiment (Kuramoto Model of oscillators with variable coupling, init frequency distribution). Output audio and system parameters of the generative model are saved in './stim-no-timbre-5/'. 

Other scripts are used to generate .csv files for various statistics of the data (mostly used in R scripts in './R/). 