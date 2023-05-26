#### LOYALISTS
library(ez)
library(dplyr)
options(scipen = 999)

setwd('/Users/nolanlem/Desktop/data/Experiment-2/scripts/R')


#
setwd('../csvs/')
datl <- read.csv("regular.csv", header = T)
datnc <- read.csv('hybrid.csv', header = T)
datc <- read.csv('fast.csv', header = T)
dat <- read.csv('all.csv', header=T)

library(plyr)

count(datl, 'subject')
count(datnc, 'subject')
count(datc, 'subject')
count(dat, "subject")

dat$coupling <- factor(dat$coupling, levels=c("none", "weak", "medium", "strong"))
dat$section <- factor(dat$beatsection, levels=c("0", "1", "2", "3", "4", "5", "6"))


datl$coupling <- factor(datl$coupling, levels=c("none", "weak", "medium", "strong"))
datl$section <- factor(datl$beatsection, levels=c("0", "1", "2", "3", "4", "5", "6"))

datnc$coupling <- factor(datnc$coupling, levels=c("none", "weak", "medium", "strong"))
datnc$section <- factor(datnc$beatsection, levels=c("0", "1", "2", "3", "4", "5", "6"))

datc$coupling <- factor(datc$coupling, levels=c("none", "weak", "medium", "strong"))
datc$section <- factor(datc$beatsection, levels=c("0", "1", "2", "3", "4", "5", "6"))

#all
anova_mx <- ezANOVA(data = dat,
                    dv=mx,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)
print(anova_mx)

########## MEAN ITI ANOVA #################

#### MX LOYALIST ANOVA
anova_mx <- ezANOVA(data = datl,
                    dv=mx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)

print(anova_mx)


# post-hoc t-test loyalists mx COUPLING 
pairwise.t.test(datl$mx,
                datl$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")

# table 
with(dat, tapply(mx, list(coupling), mean, na.rm=TRUE))
# post-hoc t-test loyalists mx SECTION 
pairwise.t.test(datl$mx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
  
  0      1      2      3      4      5     
  1 1.0000 -      -      -      -      -     
  2 0.1229 0.1717 -      -      -      -     
  3 0.0485 0.0599 1.0000 -      -      -     
  4 0.0169 0.0108 0.0897 0.2091 -      -     
  5 0.0051 0.0036 0.0339 0.0368 1.0000 -     
  6 0.0075 0.0052 0.1034 0.1127 1.0000 1.0000
with(dat, tapply(mx, list(section), mean, na.rm=TRUE))
0         1         2         3         4         5         6 
0.9139914 0.9093549 0.8772932 0.8723634 0.8564064 0.8507841 0.8489994 

## post-hoc section:coupling
datl_none <- subset(datl, coupling == 'none')
datl_weak <- subset(datl, coupling == 'weak')
datl_medium <- subset(datl, coupling == 'medium')
datl_strong <- subset(datl, coupling == 'strong')
# none
pairwise.t.test(datl_none$mx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
with(datl_none, tapply(mx, list(section), mean, na.rm=TRUE))
0         1         2         3         4         5         6 
1.2409747 1.1387812 0.9659963 0.9376873 0.8714069 0.8317501 0.7501727 


pairwise.t.test(datl_none$mx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")


### SX REGULAR TAPPERS 
anova_sx <- ezANOVA(data = datl,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)

print(anova_sx)

# POSTHOC coupling 
pairwise.t.test(datl$sx,
                datl$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# all *** except weak-medium*
with(datl, tapply(sx, list(coupling), mean, na.rm=TRUE))
none      weak    medium    strong 
0.3923244 0.2340466 0.1854333 0.1206953 
# SD decreases with increased coupling. 

# POSTHOC section 
pairwise.t.test(datl$sx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(2-6***), 1-(2**,3***,4**,5***,6***), 2-(5-6**), 3-6*
with(datl, tapply(sx, list(section), mean, na.rm=TRUE))
# 0         1         2         3         4         5         6 
# 0.4293800 0.3345389 0.2384097 0.2034494 0.1728156 0.1374558 0.1158250 
# decreases over increasing beat section 

# POSTHOC SECTION:COUPLING
datl_none <- subset(datl, coupling == 'none')
datl_weak <- subset(datl, coupling == 'weak')
datl_medium <- subset(datl, coupling == 'medium')
datl_strong <- subset(datl, coupling == 'strong')

pairwise.t.test(datl_none$sx,
                datl_none$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS 
pairwise.t.test(datl_weak$sx,
                datl_weak$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-6*, 1-3*
pairwise.t.test(datl_medium$sx,
                datl_medium$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(2*,3***,4*)
pairwise.t.test(datl_strong$sx,
                datl_strong$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
#NS 

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### MX HYBRIDS ANOVA
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

anova_mx <- ezANOVA(data = datnc,
                    dv=mx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)

print(anova_mx)


with(datnc, tapply(mx, list(coupling), mean, na.rm=TRUE))

datnc_none <- subset(datnc, coupling == 'none')
datnc_weak <- subset(datnc, coupling == 'weak')
datnc_medium <- subset(datnc, coupling == 'medium')
datnc_strong <- subset(datnc, coupling == 'strong')

pairwise.t.test(datnc_none$mx,
                datnc_none$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 1-6**, 2-6*
with(datnc_none, tapply(mx, list(section), mean, na.rm=TRUE))
# decreases from earlier to later 

pairwise.t.test(datnc_weak$mx,
                datnc_weak$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datnc_medium$mx,
                datnc_medium$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datnc_strong$mx,
                datnc_strong$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(1***,2***,3***,4***,5***,6*)
with(datnc_strong, tapply(mx, list(section), mean, na.rm=TRUE))
# strong increases nITI from beginning to end 



pairwise.t.test(datnc_none$mx,
                datnc_none$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")

#### #### #### #### #### #### #### #### #### #### #### 
##### SX HYBRID ANOVA
anova_sx <- ezANOVA(data = datnc,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)

print(anova_sx)

### POSTHOC COUPLING 
pairwise.t.test(datnc$sx,
                datnc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# all *** except weak-medium NS
with(datnc, tapply(sx, list(coupling), mean, na.rm=TRUE))
none       weak     medium     strong 
0.21454377 0.14892256 0.14265364 0.07543514 
# SD goes down with increasing coupling 
pairwise.t.test(datnc$sx,
                datnc$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(1*,2-6***), 1-(4*,5*)
with(datnc, tapply(sx, list(section), mean, na.rm=TRUE))
# 0         1         2         3         4         5         6 
# 0.2395447 0.1464995 0.1314299 0.1217379 0.1158472 0.1218662 0.1407960 
# decreases over beat section 


######### ######### ######### ######### ######### ######### ######### ######### 
######### MX FAST ANOVA ##########
anova_mx <- ezANOVA(data = datc,
                    dv=mx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)


print(anova_mx)

pairwise.t.test(datc$mx,
                datc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# all *** excpet weak-medium is NS

with(datc, tapply(mx, list(coupling), mean, na.rm=TRUE))

#### post-hoc two way interaction 
datc_none <- subset(datc, coupling == 'none')
datc_weak <- subset(datc, coupling == 'weak')
datc_medium <- subset(datc, coupling == 'medium')
datc_strong <- subset(datc, coupling == 'strong')

pairwise.t.test(datc_none$mx,
                datc_none$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-5*

with(datc_none, tapply(mx, list(section), mean, na.rm=TRUE))


with(datc_none, tapply(mx, list(section), mean, na.rm=TRUE))
# nITI decreases over beat section 

pairwise.t.test(datc_weak$mx,
                datc_weak$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datc_medium$mx,
                datc_medium$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datc_strong$mx,
                datc_strong$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
######### ######### ######### ######### ######### ######### ######### 
### SD ANOVA FOR FAST TAPPERS
######### ######### ######### ######### ######### ######### ######### 


anova_sx <- ezANOVA(data = datc,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)


print(anova_sx)

Effect DFn DFd        SSn       SSd        F           p p<.05        ges
1         coupling   3  24 0.44889838 0.5288464 6.790605 0.001786464     * 0.26380016
2          section   6  48 0.02538802 0.1324153 1.533842 0.187501853       0.01986309
3 coupling:section  18 144 0.16127445 0.5915006 2.181224 0.005927544     * 0.11405251

## POSTHOC FAST COUPLING 
pairwise.t.test(datc$sx,
                datc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# none-weak***, none-medium***, weak-strong***, medium-strong***
with(datc, tapply(sx, list(coupling), mean, na.rm=TRUE))# none      weak    medium    strong 
# 0.1353124 0.2032250 0.2116001 0.1138602 
### POSTHOC FAST COUPLING:SECTION 
datc_none <- subset(datc, coupling == 'none')
datc_weak <- subset(datc, coupling == 'weak')
datc_medium <- subset(datc, coupling == 'medium')
datc_strong <- subset(datc, coupling == 'strong')

pairwise.t.test(datc_none$sx,
                datc_none$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datc_weak$sx,
                datc_weak$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
pairwise.t.test(datc_medium$sx,
                datc_medium$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
#NS
pairwise.t.test(datc_strong$sx,
                datc_strong$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
#0-(2*,4*)
with(datc_strong, tapply(sx, list(section), mean, na.rm=TRUE))
# 0          1          2          3          4          5          6 
# 0.18730688 0.11617505 0.10603950 0.13517868 0.08326388 0.08250839 0.08654872 

## overall:
# loyalists: all significant
# noneconverts: coupling, coupling:section
# converts: coupling, coupling:section 

########## STD ITI ANOVA #################
#### STD LOYALIST ANOVA
anova_mx <- ezANOVA(data = datl,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)


##############################################################################
# post-hoc t-test loyalists mx COUPLING 
##############################################################################

pairwise.t.test(datl$sx,
                datl$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
          none               weak               medium
  weak   0.0000002784002666 -                  -     
  medium 0.0000000013065419 0.0295             -     
  strong 0.0000000000000053 0.0000041178910116 0.0039
with(dat, tapply(sx, list(coupling), mean, na.rm=TRUE))

# post hoc section
pairwise.t.test(datl$sx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
    0          1       2       3       4       5      
  1 0.20454    -       -       -       -       -      
  2 0.00004800 0.00863 -       -       -       -      
  3 0.00000289 0.00093 0.31293 -       -       -      
  4 0.00001669 0.00129 0.15115 0.84268 -       -      
  5 0.00000092 0.00026 0.00963 0.06222 1.00000 -      
  6 0.00000304 0.00017 0.00694 0.02291 0.28278 1.00000
with(dat, tapply(sx, list(section), mean, na.rm=TRUE))

######## post hoc section:coupling 
# none:section
pairwise.t.test(datl_weak$sx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")


    0     1     2     3     4     5    
  1 1.000 -     -     -     -     -    
  2 0.086 0.056 -     -     -     -    
  3 0.064 0.030 1.000 -     -     -    
  4 0.119 0.059 0.921 1.000 -     -    
  5 0.067 0.125 0.731 1.000 1.000 -    
  6 0.031 0.056 0.315 1.000 1.000 1.000

with(datl_weak, tapply(sx, list(section), mean, na.rm=TRUE))
0          1          2          3          4          5          6 
0.43496579 0.41119165 0.23580575 0.20883989 0.16945206 0.10712507 0.07094593  

pairwise.t.test(datl_medium$sx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
    0       1       2       3       4       5      
  1 1.00000 -       -       -       -       -      
  2 0.01477 0.57670 -       -       -       -      
  3 0.00054 0.26301 1.00000 -       -       -      
  4 0.04193 0.82721 1.00000 1.00000 -       -      
  5 0.07543 0.90997 1.00000 1.00000 1.00000 -      
  6 0.18249 1.00000 1.00000 1.00000 1.00000 1.00000
with(datl_medium, tapply(sx, list(section), mean, na.rm=TRUE))
0          1          2          3          4          5          6 
0.34009832 0.31663522 0.21837390 0.13493880 0.10463888 0.09033857 0.09300941 

##############################################################################
######## STD HYBRIDS ANOVA ##############################################################################
##############################################################################

anova_sx <- ezANOVA(data = datnc,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)
print(anova_sx)

Effect DFn DFd       SSn      SSd         F                   p p<.05        ges
1         coupling   3  75 1.7646527 2.043336 21.590341 0.00000000035059331     * 0.16327381
2          section   6 150 1.1510618 2.414303 11.919193 0.00000000006453584     * 0.11291181
3 coupling:section  18 450 0.2202514 4.585643  1.200766 0.25573707781485749       0.02377618

pairwise.t.test(datnc$sx,
                datnc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")

        none                 weak               medium            
  weak   0.00027              -                  -                 
  medium 0.0000000375372889   1.00000            -                 
  strong < 0.0000000000000002 0.0000000000000068 0.0000000000010986

with(datnc, tapply(sx, list(coupling), mean, na.rm=TRUE))
none       weak     medium     strong 
0.21454377 0.14892256 0.14265364 0.07543514 

## posthoc section SD HYBRID GROUP 
pairwise.t.test(datnc$sx,
                datnc$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
  
    0        1       2       3       4       5      
  1 0.00278  -       -       -       -       -      
  2 0.00022  0.84281 -       -       -       -      
  3 0.000074 0.09490 1.00000 -       -       -      
  4 0.000040 0.01024 0.31087 1.00000 -       -      
  5 0.000069 0.01692 1.00000 1.00000 1.00000 -      
  6 0.00664  1.00000 1.00000 1.00000 0.59621 1.00000

with(datnc, tapply(sx, list(section), mean, na.rm=TRUE))
0         1         2         3         4         5         6 
0.2395447 0.1464995 0.1314299 0.1217379 0.1158472 0.1218662 0.1407960 

##############################################################################
#### STD FAST GROUP ANOVA ###################################################
##############################################################################

anova_sx <- ezANOVA(data = datc,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)
print(anova_sx)

Effect DFn DFd        SSn       SSd        F           p p<.05        ges
1         coupling   3  24 0.44889838 0.5288464 6.790605 0.001786464     * 0.26380016
2          section   6  48 0.02538802 0.1324153 1.533842 0.187501853       0.01986309
3 coupling:section  18 144 0.16127445 0.5915006 2.181224 0.005927544     * 0.11405251

pairwise.t.test(datc$sx,
                datc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
        none        weak        medium     
  weak   0.00066     -           -          
  medium 0.000036877 1.00000     -          
  strong 0.92428     0.000000567 0.000000076
with(datc, tapply(sx, list(coupling), mean, na.rm=TRUE))
none      weak    medium    strong 
0.1353124 0.2032250 0.2116001 0.1138602 

datc_none = subset(datc, coupling == 'none')
datc_weak = subset(datc, coupling == 'weak')
datc_medium = subset(datc, coupling == 'medium')
datc_strong = subset(datc, coupling == 'strong')


pairwise.t.test(datc_strong$sx,
                datc_strong$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
    0     1     2     3     4     5    
  1 0.074 -     -     -     -     -    
  2 0.026 1.000 -     -     -     -    
  3 0.831 1.000 1.000 -     -     -    
  4 0.051 1.000 1.000 1.000 -     -    
  5 0.103 1.000 1.000 1.000 1.000 -    
  6 0.242 1.000 1.000 1.000 1.000 1.000
with(datc_strong, tapply(sx, list(section), mean, na.rm=TRUE))
  

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#### STD CONVERTS ANOVA
anova_mx <- ezANOVA(data = datc,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    detailed = T,
                    type=1)

print(anova_mx)

#%% t - tests 

# loyalists mx coupling 
pairwise.t.test(datl$mx,
                datl$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")

        none   weak   medium
weak   0.0013 -      -     
medium 0.0434 1.0000 -     
strong 1.0000 0.2481 0.0130

with(subset())

## NB: none-weak**, none-medium*, medium-strong*

#### NC COUPLING
pairwise.t.test(datnc$mx,
                datnc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## NB: none-(w,m,s)***, medium-strong*
## as we would expect, none is serious outlier

#### C COUPLING
pairwise.t.test(datc$mx,
                datc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## NB: none-(w,m,s)***, weak-strong***, medium-strong***
## weak medium didn't show any significant differences wrt mx 

### loyalist MX SECTION 
pairwise.t.test(datl$mx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## NB: 0-(3-6), 1-(4-6), 2-(5,6), 3-5 
## earlier to later more significant overall 

## nc, and c are not sig via anova

### look at interaction between coupling and beat section for MX 
### subset by coupling and look at section 
########## loyalist
datln = subset(datl, coupling == 'none')
datlw = subset(datl, coupling == 'weak')
datlm = subset(datl, coupling == 'medium')
datls = subset(datl, coupling == 'strong')

# loyalist none section
pairwise.t.test(datln$mx,
                datln$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NB: 3-6, 4-6

# loyalist weak section
pairwise.t.test(datlw$mx,
                datlw$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
# loyalist med section 
pairwise.t.test(datlm$mx,
                datlm$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
# loyalist strong section
pairwise.t.test(datls$mx,
                datls$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")

# overall: # only none has significant from mid beats (3,4) to last bs

# NS
# NONECONVERTS
datncn = subset(datnc, coupling == 'none')
datncw = subset(datnc, coupling == 'weak')
datncm = subset(datnc, coupling == 'medium')
datncs = subset(datnc, coupling == 'strong')

# nc none section
pairwise.t.test(datncn$mx,
                datncn$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NB: 0-6, 1-6, 2-6, 

# nc weak section
pairwise.t.test(datncw$mx,
                datncw$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
# nc med section 
pairwise.t.test(datncm$mx,
                datncm$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

# nc strong section
pairwise.t.test(datncs$mx,
                datncs$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(1,6)
# diff comes from following... 
## overall: none -> early beat segs significant with almost all other beat segs 
#           strong _> first beat seg signif with all others


# CONVERTS
datcn = subset(datc, coupling == 'none')
datcw = subset(datc, coupling == 'weak')
datcm = subset(datc, coupling == 'medium')
datcs = subset(datc, coupling == 'strong')

# c none section
pairwise.t.test(datcn$mx,
                datcn$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NB: 0-5*

# c weak section
pairwise.t.test(datcw$mx,
                datcw$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS
# c med section 
pairwise.t.test(datcm$mx,
                datcm$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

# c strong section
pairwise.t.test(datcs$mx,
                datcs$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

## overall: only none has significance with one other later beat seg

######### SX T-TESTS  #########
# loyalists: all
# nc: coupling, section
# c: coupling, coupling:section 

# loyalists coupling sx
pairwise.t.test(datl$sx,
                datl$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## none-(w,m,s)***, weak-(m*,s***), medium-(s**)

# noneconverts coupling sx 
pairwise.t.test(datnc$sx,
                datnc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## none-(w**,m***, s***), weak-strong***, medium-strong***

# converts coupling sx 
pairwise.t.test(datc$sx,
                datc$coupling,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## none-(w***,m***), weak-strong***, medium-strong***

# loyalists sx section
pairwise.t.test(datl$sx,
                datl$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## 0-(2-6), 1-(2-6), 2(5,6), 3-6 

# noneconverts sx section
pairwise.t.test(datnc$sx,
                datnc$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## 0-(1-6), 1-(4,5)


## loyal: interaction btw section coupling
# loyalists sx section none
pairwise.t.test(datln$sx,
                datln$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

pairwise.t.test(datlw$sx,
                datlw$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-6, 1-(2-4,6)

pairwise.t.test(datlm$sx,
                datlm$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## 0-(2-4)

pairwise.t.test(datls$sx,
                datls$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

# interaction is btw weak and medium beat sections sig

### Converts interaction between coupling and section 
pairwise.t.test(datcn$sx,
                datcn$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

pairwise.t.test(datcw$sx,
                datcw$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# NS

pairwise.t.test(datcm$sx,
                datcm$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
## NS

pairwise.t.test(datcs$sx,
                datcs$section,
                paired = TRUE,
                p.adjust.method = "bonferroni")
# 0-(2,4)

## interaction is just because strong sx has significant from 0-(2,4)

#%% BETWEEN GROUPS

anova_mx <- ezANOVA(data = dat,
                    dv=sx,
                    wid = subject,
                    within = .(coupling, section),
                    between = subcat,
                    detailed = T, type=1)
print(anova_mx)


#####




  
  
  
  
