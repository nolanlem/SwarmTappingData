install.packages('lme4')
library(lme4)
library(dplyr)
library(tidyr)
#


df = read.csv('../csvs/subjects-disp-centroid-d-s-R.csv', header=TRUE, stringsAsFactors= TRUE)
#which(is.na(df)==T)

# remove repeats bb, cc
df[!grepl("bb_kura-A2_2020-08-08_23h32.56.541.csv", df$colum),]
df[!grepl("cc_kura-B1_2020-08-09_00h01.21.424.csv", df$column),]


head(df)
summary(df)

## don't have to do this anymore because fixed df, now k = ['1','3d','3s']
# combine k cluster 1 & 2
df$k.means.grp[(df$k.means.grp == 1) | (df$k.means.grp == 2)] <- "1" 
# combine k cluster remove k clusters 4 and 5
df<-df[!(df$k.means.grp=="4" | df$k.means.grp =="5"),]
df<-df[!(df$k==4 | df$k ==5),]

df<-df[!(df$k=="4" | df$k =="5"),]
df<-df[!(df$k=="4" | df$k =="5"),]

df_k.3ds <- subset(df, df$k == '3d' | df$k == '3s')
# create column to tag k=3 group as dense or sparse
df_k.3ds <- df_k.3ds %>% 
  mutate(ktag = ifelse(k == '3d', 'd', 's')) %>% 
  fill(k)

df_k.3ds <- mutate(df_k.3ds, k.tag = ifelse(k == '3d', "d", "s"))
summary(df_k.3ds)

df <- na.omit(df) # remove nans 
df$k.means.grp = factor(df$k.means.grp) # recreate factor after having deleted the tag 
df$k = factor(df$k) # recreate factor after having deleted the tag 




boxplot(df$dist.cent ~ df$stim.mean.tempo, col=c("white", "lightgray"), df)
boxplot(df$dist.cent ~ df$k, col=c("white", "lightgray"), df)
boxplot(df$stim.mean.tempo ~ df$k.means.grp, col=c("white", "lightgray"), df)


df.k.12 = subset(df, k.means.grp=='1')
df.k.3d = subset(df, k.means.grp=='3d')
df.k.3s = subset(df, k.means.grp=='3s')

df.k.3 = subset(df, k=='3')


df.reg = subset(df, subject.cat == 'loyalist')
df.hybrid = subset(df, subject.cat == 'noneconvert')
df.fast = subset(df, subject.cat == 'convert')

# how many trial in reg, hybrid an fast group were in k=3d and k=3s? 
df.reg.3d = subset(df, (subject.cat == 'loyalist') & (k.means.grp == '3d'))
df.reg.3s = subset(df, (subject.cat == 'loyalist') & (k.means.grp == '3s'))
# by coupling 
df.reg.3d.s = subset(df.reg.3d, coupling == 'strong')
df.reg.3d.m = subset(df.reg.3d, coupling == 'medium')
df.reg.3d.w = subset(df.reg.3d, coupling == 'weak')
df.reg.3d.n = subset(df.reg.3d, coupling == 'none')

df.reg.3s.s = subset(df.reg.3s, coupling == 'strong')
df.reg.3s.m = subset(df.reg.3s, coupling == 'medium')
df.reg.3s.w = subset(df.reg.3s, coupling == 'weak')
df.reg.3s.n = subset(df.reg.3s, coupling == 'none')

df.hybrid.3d.s = subset(df.hybrid.3d, coupling == 'strong')
df.hybrid.3d.m = subset(df.hybrid.3d, coupling == 'medium')
df.hybrid.3d.w = subset(df.hybrid.3d, coupling == 'weak')
df.hybrid.3d.n = subset(df.hybrid.3d, coupling == 'none')

df.hybrid.3s.s = subset(df.hybrid.3s, coupling == 'strong')
df.hybrid.3s.m = subset(df.hybrid.3s, coupling == 'medium')
df.hybrid.3s.w = subset(df.hybrid.3s, coupling == 'weak')
df.hybrid.3s.n = subset(df.hybrid.3s, coupling == 'none')

df.fast.3d.s = subset(df.fast.3d, coupling == 'strong')
df.fast.3d.m = subset(df.fast.3d, coupling == 'medium')
df.fast.3d.w = subset(df.fast.3d, coupling == 'weak')
df.fast.3d.n = subset(df.fast.3d, coupling == 'none')

df.fast.3s.s = subset(df.fast.3s, coupling == 'strong')
df.fast.3s.m = subset(df.fast.3s, coupling == 'medium')
df.fast.3s.w = subset(df.fast.3s, coupling == 'weak')
df.fast.3s.n = subset(df.fast.3s, coupling == 'none')

summary(df.fast.3s.n)

count(df.fast.3s.n)

df.hybrid.3d = subset(df, (subject.cat == 'noneconvert') & (k.means.grp == '3d'))
df.hybrid.3s = subset(df, (subject.cat == 'noneconvert') & (k.means.grp == '3s'))

df.fast.3d = subset(df, (subject.cat == 'convert') & (k.means.grp == '3d'))
df.fast.3s = subset(df, (subject.cat == 'convert') & (k.means.grp == '3s'))


df$sync.type <- factor(df$coupling, levels=c("strong", "medium", "weak", "none"))
df.k.3d$sync.type <- factor(df.k.3d$coupling, levels=c("strong", "medium", "weak", "none"))
df.k.3s$sync.type <- factor(df.k.3s$coupling, levels=c("strong", "medium", "weak", "none"))

df$sync.type <- factor(df$coupling, levels=c("strong", "medium", "weak", "none"))

### effects of k group on mean iti, should expect to see sig differences here
boxplot(df$tap.mean.iti ~ df$k.means.grp, col=c("white", "lightgray"), df)
df.lm <- lm(df$tap.mean.iti ~ df$k.means.grp, data = df)
summary(df.lm) # F-statistic: 544.7 on 2 and 1992 DF,  p-value: < 2.2e-16

boxplot(df.k3$tap.mean.iti ~ df$sync.type, col=c("white", "lightgray"), df)


### for k = 3 DENSE, regression with base tempo 
par(mfrow=c(1,2)) 
boxplot(df.k.3d$tap.mean.iti ~ df.k.3d$base.tempo, col=c("white", "lightgray"), df.k.3d, main='k=3 dense')
df.lm <- lm(df.k.3d$tap.mean.iti ~ df.k.3d$base.tempo, data = df.k.3d)
summary(df.lm) # F-statistic: 78.47 on 1 and 299 DF,  p-value: < 2.2e-16

### for k = 3 SPARSE, regression with base tempo 
boxplot(df.k.3s$tap.mean.iti ~ df.k.3s$base.tempo, col=c("white", "lightgray"), df.k.3s,
        main='k=3 sparse')
df.lm <- lm(df.k.3s$tap.mean.iti ~ df.k.3s$base.tempo, data = df.k.3s)
summary(df.lm) # F-statistic: 78.47 on 1 and 299 DF,  p-value: < 2.2e-16

library(ggplot2)
library(tidyverse)
library(ggpubr)
fun_mean <- function(x){return(round(data.frame(y=mean(x),label=mean(x,na.rm=T)),digit=2))}


####### Take all k=3d, use R as explanatory variable, and base tempo, and 
### group (regular, hybrid, and  fast) try to make linear model (lm) . 
## See if it produces interactions between coupling or tempo 
Iti ~ lm(R + tempo*group) 
(R + tempo) * group 
(R*group) 
(tempo*group) 

boxplot(df.k.3d$tap.mean.iti ~ df.k.3d$base.tempo * df.k.3d$subject.cat, las=2)
boxplot(df.k.3s$tap.mean.iti ~ df.k.3s$base.tempo * df.k.3s$subject.cat, las=2)


df.k.3 <- subset(df, (df$k == '3d') | (df$k == '3s'))
df.lm <-lm(df.k.3$tap.mean.iti ~ df.k.3$base.tempo * df.k.3$subject.cat, data = df.k.3)


boxplot(df_k.3ds$tap.mean.iti ~  (df_k.3ds$R.model+df_k.3ds$k.tag), las=2)
df.lm <-lm(df_k.3ds$tap.mean.iti ~  (df_k.3ds$R.model+df_k.3ds$k.tag), data = df_k.3ds)
summary(df.lm)

df.k.3d = subset(df, k.means.grp=='3d')
#fruit_words <- c("apple", "orange", "banana", "pappels", "orong", "bernaner")


abc <- c(loyalist="A", noneconvert="hybrid", convert="fast")
df.k.3d$subject.cat.rev <- as.character(abc[df.k.3d$subject.cat])
df.k.3s$subject.cat.rev <- as.character(abc[df.k.3s$subject.cat])

df.k.3d$subject.cat.rev

print(df.k.3d)

library(broom)
library(clipr)
library(magrittr)
# iti + (R * tempo)*group
df.lm <-lm(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat.rev, data = df.k.3d) %>%
tidy() %>%
  write_clip()

df.lm <-lm(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat.rev, data = df.k.3d)
summary(df.lm)

df.lm <-lm(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat.rev, data = df.k.3d)
summary(df.lm)

Residual standard error: 0.1149 on 289 degrees of freedom
Multiple R-squared:  0.6088,	Adjusted R-squared:  0.5939 
F-statistic: 40.88 on 11 and 289 DF,  p-value: < 0.00000000000000022

#SPARSE: iti + (R * tempo)*group
df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat.rev, data = df.k.3s) %>%
  tidy() %>%
  write_clip()


df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat.rev, data = df.k.3s)
summary(df.lm)

  

Residual standard error: 0.1132 on 195 degrees of freedom
Multiple R-squared:  0.5586,	Adjusted R-squared:  0.5337 
F-statistic: 22.43 on 11 and 195 DF,  p-value: < 0.00000000000000022

# iti + (R*group)
boxplot(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model * df.k.3d$subject.cat), las=2)
df.lm <-lm(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model * df.k.3d$subject.cat), data = df.k.3d)
summary(df.lm)
Estimate Std. Error t value Pr(>|t|)    
(Intercept)                                     0.20718    0.01975  10.493   <2e-16 ***
df.k.3d$R.model                                 0.04616    0.07125   0.648   0.5177    
df.k.3d$subject.catloyalist                     0.18297    0.11818   1.548   0.1227    
df.k.3d$subject.catnoneconvert                  0.27932    0.02655  10.520   <2e-16 ***
df.k.3d$R.model:df.k.3d$subject.catloyalist     2.68362    1.54099   1.741   0.0827 .  
df.k.3d$R.model:df.k.3d$subject.catnoneconvert -0.28094    0.18519  -1.517   0.1304    

# iti + (tempo*group)
df.lm <-lm(df.k.3d$tap.mean.iti ~ (df.k.3d$base.tempo * df.k.3d$subject.cat), data = df.k.3d)
summary(df.lm)

df.k.3d$base.tempo                                   
df.k.3d$subject.catloyalist                       ***
df.k.3d$subject.catnoneconvert                    ***
df.k.3d$base.tempo:df.k.3d$subject.catloyalist    ***
df.k.3d$base.tempo:df.k.3d$subject.catnoneconvert ***
  
  
### for k = 3 sparse 
# iti + (R * tempo)*group
df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat, data = df.k.3s)
summary(df.lm)
Estimate Std. Error t value Pr(>|t|)    
(Intercept)                                                        0.3519163  0.1116797   3.151 0.001887 ** 
df.k.3s$R.model                                                    0.0130038  0.3463884   0.038 0.970092    
df.k.3s$base.tempo                                                -0.0001267  0.0014507  -0.087 0.930499    
df.k.3s$subject.catloyalist                                        0.6479018  0.2024693   3.200 0.001608 ** 
df.k.3s$subject.catnoneconvert                                     0.6407315  0.1466642   4.369 2.04e-05 ***
df.k.3s$R.model:df.k.3s$base.tempo                                -0.0011902  0.0043505  -0.274 0.784704    
df.k.3s$R.model:df.k.3s$subject.catloyalist                        0.8732369  1.3628430   0.641 0.522451    
df.k.3s$R.model:df.k.3s$subject.catnoneconvert                    -1.1744626  0.6387199  -1.839 0.067493 .  
df.k.3s$base.tempo:df.k.3s$subject.catloyalist                    -0.0066783  0.0027589  -2.421 0.016423 *  
df.k.3s$base.tempo:df.k.3s$subject.catnoneconvert                 -0.0065624  0.0019541  -3.358 0.000946 ***
df.k.3s$R.model:df.k.3s$base.tempo:df.k.3s$subject.catloyalist    -0.0083612  0.0163765  -0.511 0.610245    
df.k.3s$R.model:df.k.3s$base.tempo:df.k.3s$subject.catnoneconvert  0.0143143  0.0079310   1.805 0.072666 .  

#### dispersion , k=3 dense
boxplot(df.k.3d$tap.disp ~  df.k.3d$subject.cat, las=2)
boxplot(df.k.3s$tap.disp ~  df.k.3s$subject.cat, las=2)

boxplot(df_k.3ds$tap.disp ~  df_k.3ds$subject.cat + df_k.3ds$ktag, las=2)


df.lm <-lm(df.k.3d$tap.disp ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat.rev, data = df.k.3d) %>%
  tidy() %>%
  write_clip()

df.lm <-lm(df.k.3s$tap.disp ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat.rev, data = df.k.3s) %>%
  tidy() %>%
  write_clip()

df.lm <-lm(df.k.3d$tap.disp ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat.rev, data = df.k.3d) 
summary(df.lm)
  
boxplot(df.k.3d$dist.cent ~  df.k.3d$subject.cat, las=2)
boxplot(df.k.3s$dist.cent ~  df.k.3s$subject.cat, las=2)


### centroid dist differences between sparse and dense
df.lm <-lm(df.k.3d$dist.cent ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat, data = df.k.3d)
summary(df.lm)

df.lm <-lm(df.k.3s$dist.cent ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat, data = df.k.3s)
summary(df.lm)


df.lm <-lm(df.k.3d$tap.disp ~ (df.k.3d$R.model * df.k.3d$base.tempo)*df.k.3d$subject.cat, data = df.k.3d)
summary(df.lm)

(Intercept)                                                        0.0454336  0.0161727   2.809  0.00531 **
df.k.3d$R.model                                                    0.0412489  0.0657009   0.628  0.53062   
df.k.3d$base.tempo                                                -0.0000945  0.0001954  -0.484  0.62900   
df.k.3d$subject.catloyalist                                        0.0659095  0.0944175   0.698  0.48571   
df.k.3d$subject.catnoneconvert                                     0.0827922  0.0355530   2.329  0.02058 * 
df.k.3d$R.model:df.k.3d$base.tempo                                 0.0002143  0.0007182   0.298  0.76560   
df.k.3d$R.model:df.k.3d$subject.catloyalist                        0.0089326  1.3638565   0.007  0.99478   
df.k.3d$R.model:df.k.3d$subject.catnoneconvert                    -0.5267798  0.4101015  -1.285  0.20001   
df.k.3d$base.tempo:df.k.3d$subject.catloyalist                    -0.0005949  0.0012259  -0.485  0.62786   
df.k.3d$base.tempo:df.k.3d$subject.catnoneconvert                 -0.0007537  0.0004478  -1.683  0.09343 . 
df.k.3d$R.model:df.k.3d$base.tempo:df.k.3d$subject.catloyalist    -0.0035777  0.0191030  -0.187  0.85157   
df.k.3d$R.model:df.k.3d$base.tempo:df.k.3d$subject.catnoneconvert  0.0049834  0.0052232   0.954  0.34085

# dispersion spares
df.lm <-lm(df.k.3s$tap.disp ~ (df.k.3s$R.model * df.k.3s$base.tempo)*df.k.3s$subject.cat, data = df.k.3s)
summary(df.lm)

# iti + (R*group)
df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$R.model * df.k.3s$subject.cat), data = df.k.3s)
summary(df.lm)
Estimate Std. Error t value Pr(>|t|)    
(Intercept)                                     0.34148    0.02905  11.756  < 2e-16 ***
df.k.3s$R.model                                -0.09383    0.07670  -1.223   0.2227    
df.k.3s$subject.catloyalist                     0.26154    0.04896   5.342 2.51e-07 ***
df.k.3s$subject.catnoneconvert                  0.19263    0.03695   5.213 4.66e-07 ***
df.k.3s$R.model:df.k.3s$subject.catloyalist    -0.23532    0.26954  -0.873   0.3837    
df.k.3s$R.model:df.k.3s$subject.catnoneconvert -0.23250    0.12312  -1.888   0.0604 .  


# iti + (tempo*group)
df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$base.tempo * df.k.3s$subject.cat), data = df.k.3s)
summary(df.lm)
(Intercept)                                       9.07e-12 ***
df.k.3s$base.tempo                                   0.254    
df.k.3s$subject.catloyalist                       2.58e-09 ***
df.k.3s$subject.catnoneconvert                    1.01e-09 ***
df.k.3s$base.tempo:df.k.3s$subject.catloyalist    1.89e-05 ***
df.k.3s$base.tempo:df.k.3s$subject.catnoneconvert 4.73e-06 ***

  
# iti + (tempo*group)
df.lm <-lm(df.k.3s$tap.mean.iti ~ (df.k.3s$base.tempo * df.k.3s$subject.cat), data = df.k.3s)
summary(df.lm)





boxplot(df.k.3d$tap.mean.iti ~ (df.k.3d$R.model + df.k.3d$base.tempo)* df.k.3d$subject.cat, las=2)



df.k.3d$R.model = round(fun_mean, digit=2)
p <- boxplot(df.k.3d$tap.mean.iti ~ df.k.3d$R.model, col=c("white", "lightgray"), df.k.3d,
        main='k=3 dense', las=2) + stat_summary(fun.data= function(x) data.frame(y=1, label = paste("Mean=", round(mean(x), 2))), geom="text")



boxplot(df_ds$tap.mean.iti ~ df_ds$base.tempo, col=c("white", "lightgray"), df.k.3d, main='k=3 dense vs sparse')


df_ds <- subset(df, (df$k.means.grp == '3d') | (df$k.means.grp == '3s'))
df_ds.lm <-lm(df_ds$tap.mean.iti ~ df_ds$base.tempo * df_ds$k.means.grp, data = df_ds)
summary(df_ds.lm)
# F-statistic: 74.39 on 2 and 505 DF,  p-value: < 2.2e-16
Estimate Std. Error t value Pr(>|t|)    
(Intercept)          0.7184768  0.0289119  24.851   <2e-16 ***
  df_ds$base.tempo    -0.0042392  0.0003491 -12.145   <2e-16 ***
  df_ds$k.means.grp3s  0.0208086  0.0138872   1.498    0.135   

# so 3s and 3d responses are not significantly different ito mean tap iti, 
#######  what about in terms of dispersion? 

df_ds.lm <-lm(df_ds$tap.disp ~ df_ds$base.tempo + df_ds$k.means.grp, data = df_ds)
summary(df_ds.lm)
 (Intercept)          0.0921113  0.0154662   5.956 4.86e-09 ***
  df_ds$base.tempo    -0.0004769  0.0001867  -2.554   0.0109 *  
  df_ds$k.means.grp3s  0.1660430  0.0074288  22.351  < 2e-16 ***
# *** F-statistic: 251.6 on 2 and 505 DF,  p-value: < 2.2e-16
# Dispersion is meaninfully different (which makes sense because I used dispersion to split them)

###### what about in terms of centroid differences 
df_ds.lm <-lm(df_ds$dist.cent ~ df_ds$base.tempo * df_ds$k.means.grp, data = df_ds)
summary(df_ds.lm)
# NB: not significantly different in terms of centroid difference 
## (Intercept)          1.2212902  0.0406971  30.009   <2e-16 ***
  df_ds$base.tempo    -0.0078321  0.0004914 -15.940   <2e-16 ***
  df_ds$k.means.grp3s -0.0288263  0.0195479  -1.475    0.141


##### what about just for NONE coupling condition within k=3?   
boxplot(df_ds.none$tap.mean.iti ~ df_ds.none$base.tempo + df_ds.none$k.means.grp, col=c("white", "lightgray"), df_ds.none, main='k=3 none')
  
df_ds.none <- subset(df_ds, df_ds$sync.type == 'none')
df_ds.lm <-lm(df_ds.none$tap.mean.iti ~ df_ds.none$base.tempo + df_ds.none$k.means.grp, data = df_ds.none)
summary(df_ds.lm)
  (Intercept)               0.7372048  0.0447156  16.487  < 2e-16 ***
  df_ds.none$base.tempo    -0.0042945  0.0005879  -7.305 2.06e-12 ***
  df_ds.none$k.means.grp3s  0.0485526  0.0208807   2.325   0.0207 * 
## NB: * p == 0.0207
  
##### what about just for WEAK coupling condition within k=3?  
df_ds.weak <- subset(df_ds, df_ds$sync.type == 'weak')
boxplot(df_ds.weak$tap.mean.iti ~ df_ds.weak$base.tempo + df_ds.weak$k.means.grp, col=c("white", "lightgray"), df_ds.weak, main='k=3 weak')

df_ds.lm <-lm(df_ds.weak$tap.mean.iti ~ df_ds.weak$base.tempo + df_ds.weak$k.means.grp, data = df_ds.weak)
summary(df_ds.lm)

(Intercept)               0.3271410  0.0493054   6.635 2.65e-09 ***
df_ds.weak$base.tempo    -0.0009047  0.0004775  -1.895  0.06145 .  
df_ds.weak$k.means.grp3s  0.0759640  0.0191559   3.966  0.00015 ***


##### what about just for MEDIUM coupling condition within k=3?  
df_ds.medium <- subset(df_ds, df_ds$sync.type == 'medium')
boxplot(df_ds.medium$tap.mean.iti ~ df_ds.medium$base.tempo + df_ds.medium$k.means.grp, col=c("white", "lightgray"), df_ds.medium, main='k=3 medium')
df_ds.lm <-lm(df_ds.medium$tap.mean.iti ~ df_ds.medium$base.tempo + df_ds.medium$k.means.grp, data = df_ds.weak)
summary(df_ds.lm)
Estimate Std. Error t value Pr(>|t|)    
(Intercept)                 0.3577506  0.0557628   6.416 9.56e-09 ***
  df_ds.medium$base.tempo    -0.0012344  0.0005278  -2.339 0.021887 *  
  df_ds.medium$k.means.grp3s  0.0790134  0.0203638   3.880 0.000215 ***
# ***, p = 0.000215, same 
  
  
### k=3 dense and sparse for REGULAR, HYBRID, AND FAST 
df.reg.3ds <- subset(df.reg, (df.reg$k.means.grp == '3d') | (df.reg$k.means.grp == '3s')) 
df.hybrid.3ds <- subset(df.hybrid, (df.hybrid$k.means.grp == '3d') | (df.hybrid$k.means.grp == '3s')) 
df.fast.3ds <- subset(df.fast, (df.fast$k.means.grp == '3d') | (df.fast$k.means.grp == '3s')) 

# comparing 3d and 3s within REGULAR group 
boxplot(df.reg.3ds$tap.mean.iti ~ df.reg.3ds$base.tempo, col=c("white", "lightgray"), df.reg.3ds,
        main='regular k=3 d,s')
df.lm <-lm(df.reg.3ds$tap.mean.iti ~ df.reg.3ds$base.tempo + df.reg.3ds$k.means.grp, data = df.reg.3ds)
summary(df.lm)
# NB: NS wrt k means grp (3d vs 3s)

# comparing 3d and 3s within HYBRID GROUP 
boxplot(df.hybrid.3ds$tap.mean.iti ~ df.hybrid.3ds$base.tempo, col=c("white", "lightgray"), df.hybrid.3ds,
        main='hybrid k=3 d,s')
df.lm <-lm(df.hybrid.3ds$tap.mean.iti ~ df.hybrid.3ds$base.tempo + df.hybrid.3ds$k.means.grp, data = df.hybrid.3ds)
summary(df.lm)
# NB: NS wrt k means grp (3d vs 3s)

# comparing 3d and 3s within FAST GROUP 
boxplot(df.fast.3ds$tap.mean.iti ~ df.fast.3ds$base.tempo, col=c("white", "lightgray"), df.fast.3ds,
        main='fast k=3 d,s')
df.lm <-lm(df.fast.3ds$tap.mean.iti ~ df.fast.3ds$base.tempo + df.fast.3ds$k.means.grp, data = df.fast.3ds)
summary(df.lm)
Estimate Std. Error t value Pr(>|t|)    
(Intercept)                0.2461592  0.0201644   12.21   <2e-16 ***
  df.fast.3ds$base.tempo    -0.0003212  0.0002155   -1.49    0.138    
df.fast.3ds$k.means.grp3s  0.0907216  0.0086973   10.43   <2e-16 ***
# NB: F-statistic: 56.78 on 2 and 218 DF,  p-value: < 2.2e-16
 ### *** significant between 3d and 3s. 

### k=3 dense for REGULAR, HYBRID, AND FAST 
df.reg.k3d <- subset(df.reg, df.reg$k.means.grp == '3d') 
df.hybrid.k3d <- subset(df.hybrid, df.hybrid$k.means.grp == '3d') 
df.fast.k3d <- subset(df.fast, df.fast$k.means.grp == '3d') 

par(mfrow-c(1,3))

boxplot(df.reg.k3d$tap.mean.iti ~ df.reg.k3d$base.tempo, col=c("white", "lightgray"), df.reg.k3d,
        main='regular k=3 dense')
boxplot(df.hybrid.k3d$tap.mean.iti ~ df.hybrid.k3d$base.tempo, col=c("white", "lightgray"), df.hybrid.k3d,
        main='hybrid k=3 dense')
boxplot(df.fast.k3d$tap.mean.iti ~ df.fast.k3d$base.tempo, col=c("white", "lightgray"), df.fast.k3d,
        main='fast k=3 dense')

### k=3 sparse for REGULAR, HYBRID, AND FAST 
df.reg.k3s <- subset(df.reg, df.reg$k.means.grp == '3s') 
df.hybrid.k3s <- subset(df.hybrid, df.hybrid$k.means.grp == '3s') 
df.fast.k3s <- subset(df.fast, df.fast$k.means.grp == '3s') 

par(mfrow-c(1,3))

boxplot(df.reg.k3s$tap.mean.iti ~ df.reg.k3s$base.tempo, col=c("white", "lightgray"), df.reg.k3s,
        main='regular k=3 sparse')
boxplot(df.hybrid.k3s$tap.mean.iti ~ df.hybrid.k3s$base.tempo, col=c("white", "lightgray"), df.hybrid.k3s,
        main='hybrid k=3 sparse')
boxplot(df.fast.k3s$tap.mean.iti ~ df.fast.k3s$base.tempo, col=c("white", "lightgray"), df.fast.k3s,
        main='fast k=3 sparse')


# tap mean iti by coupling and k type
boxplot(df$tap.mean.iti ~ df$sync.type*df$k, col=c("white", "lightgray"), df, las=2)
df.lm <- lm(df$tap.mean.iti ~ df$sync.type*df$k, data = df)
summary(df.lm) # *medium(p=0.0203), *** medium:k (p<<0.001), * weak:(p=0.248) F-statistic: 6.685 on 6 and 2027 DF,  p-value: 5.089e-07

# dispersion and k grp
boxplot(df$tap.disp ~ df$k, col=c("white", "lightgray"), df, las=2) # disp higher in k==3 group
df.lm <- lm(df$tap.disp ~ df$k, data = df)
summary(df.lm) # *** F(25.34), p=5.22e-07
# centroid difference and k group
boxplot(df$dist.cent ~ df$k, col=c("white", "lightgray"), df, las=2) # centroid diff larger in k==3 group, expected
df.lm <- lm(df$dist.cent ~ df$k, data = df)
summary(df.lm) # *** F-statistic:  2411 on 1 and 2033 DF,  p-value: < 2.2e-16

boxplot(df$tap.disp ~ df$sync.type*df$k, col=c("white", "lightgray"), df, las=2)
boxplot(df$dist.cent ~ df$sync.type*df$k, col=c("white", "lightgray"), df, las=2)


lme.model = lmer(df.k.3$tap.mean.iti ~ df$k + df$sync.type + (1|subjects)  , data=df.k.3)
df.lm = lm(df.k.3$tap.mean.iti ~ df.k.3$k*df.k.3$sync.type , data=df.k.3)
summary(df.lm)

######## all taps, all k ########
boxplot(df$tap.mean.iti ~ df$sync.type, col=c("white", "lightgray"), df)

##### all K: Effects of base tempo on mean tap ITI
# all taps mean iti, base tempo
boxplot(df$tap.mean.iti ~ df$base.tempo, col=c("white", "lightgray"), df)
# linear regression analysis of effects of base tempo on tap mean iti, 
df.lm <- lm(df$tap.mean.iti ~ df$base.tempo, data = df)
summary(df.lm) # *** F-statistic:    76 on 1 and 2032 DF,  p-value: < 2.2e-16


boxplot(df.k.3$tap.mean.iti ~ df.k.3$base.tempo*df.k.3$sync.type, col=c("white", "lightgray"), df.k.3)

boxplot(df.k.3$tap.mean.iti ~ df.k.3$base.tempo, col=c("white", "lightgray"), df.k.3, ylim=c(0,2.0))

boxplot(df.k.12$tap.mean.iti ~ df.k.12$base.tempo, col=c("white", "lightgray"), df.k.12, ylim=c(0,2.0))

##### K==3: Effects of base tempo on mean tap ITI
boxplot(df.k.3$tap.mean.iti ~ df.k.3$base.tempo, col=c("white", "lightgray"), df.k.3)
df.lm <- lm(df.k.3$tap.mean.iti ~ df.k.3$base.tempo, data = df.k.3)
summary(df.lm) # *** F-statistic: 146.2 on 1 and 506 DF,  p-value: < 2.2e-16

boxplot(df.k.12$tap.mean.iti ~ df.k.12$sync.type, col=c("white", "lightgray"), df.k.12)

##### K==3: Effects of coupling on mean tap ITI
boxplot(df.k.3$tap.mean.iti ~ df.k.3$sync.type, col=c("white", "lightgray"), df.k.3)
# NB: appears that mean tap ITI for none coupling is higher than for medium and weak for k==3 group, need to validate
# same but only for 'dense' k=3 tapping
df.lm <- lm(df.k.3$tap.mean.iti ~ df.k.3$sync.type, data = df.k.3)
summary(df.lm) # *** F-statistic:    47 on 2 and 505 DF,  p-value: < 2.2e-16

##### K==3 WEAK: Effects of base tempo on mean tap ITI
## what about only for weak, k==3 
df.k.3.weak <- subset(df.k.3, df.k.3$coupling == 'weak')
boxplot(df.k.3.weak$tap.mean.iti ~ df.k.3.weak$base.tempo, col=c("white", "lightgray"), df.k.3.weak)
# check l regression for effects of base tempo on tap mean iti
df.lm <- lm(df.k.3.weak$tap.mean.iti ~ df.k.3.weak$base.tempo, data = df.k.3.weak)
summary(df.lm) # * F-statistic: 5.524 on 1 and 88 DF,  p-value: 0.02099
# NB: mean tap iti goes down with tempo, we should expect this for weakly coupled stim
# since subjects may be able to loosely feel the beat 

###### K==3 NONE: Effects of base tempo on mean tap ITI
## what about only for none, k==3 
df.k.3.none <- subset(df.k.3, df.k.3$coupling == 'none')
boxplot(df.k.3.none$tap.mean.iti ~ df.k.3.none$base.tempo, col=c("white", "lightgray"), df.k.3.none)
df.lm <- lm(df.k.3.none$tap.mean.iti ~ df.k.3.none$base.tempo, data = df.k.3.none)
summary(df.lm) # *** F-statistic: 58.68 on 1 and 334 DF,  p-value: 2.03e-13, 
### NB: even with no coupling, faster tap rate when tempo increases 
##     meaning that onsets that are  uncoupled but are occuring at a faster rate, 'fast' tapping 
##     subjects tend to tap at a quicker rate

### EFFECST of BASE TEMPO on DISPERSION in k == 3 group  
## what about dispersion in these groups 
boxplot(df.k.3$tap.disp ~ df.k.3$base.tempo, col=c("white", "lightgray"), df.k.3)
df.lm <- lm(df.k.3$tap.disp ~ df.k.3$base.tempo, data = df.k.3)
summary(df.lm) # NS, p = 0.181 
# base tempo effect on dispersion for weak coupling? 
boxplot(df.k.3.weak$tap.disp ~ df.k.3.weak$base.tempo, col=c("white", "lightgray"), df.k.3.weak)
df.lm <- lm(df.k.3.weak$tap.disp ~ df.k.3.weak$base.tempo, data = df.k.3.weak)
summary(df.lm) # F(6.763) p = 0.011 *
# NB: dispersion goes down with increasing stim base tempo
# meaning: upon faster tempos, we see more tap to tap consistency (less variability)

# base tempo effect on dispersion for none coupling? 
boxplot(df.k.3.none$tap.disp ~ df.k.3.none$base.tempo, col=c("white", "lightgray"), df.k.3.none)
df.lm <- lm(df.k.3.none$tap.disp ~ df.k.3.none$base.tempo, data = df.k.3.none)
summary(df.lm) # F(6.07) p = 0.014 *

###### CENTROID DIFFERENCES ~ BASE TEMPO #### #### ####
# what about effects of tempo on centroid diff with fast k = 3 group? 
boxplot(df.k.3$dist.cent ~ df.k.3$base.tempo, col=c("white", "lightgray"), df.k.3)
df.lm <- lm(df.k.3$dist.cent ~ df.k.3$base.tempo, data = df.k.3)
summary(df.lm) #  *** F(255.1), p < 2.2e-16 

# what about effects of tempo on centroid diff with fast k = 3 group only for weak coupling? 
boxplot(df.k.3.weak$dist.cent ~ df.k.3.weak$base.tempo, col=c("white", "lightgray"), df.k.3.weak)
df.lm <- lm(df.k.3.weak$dist.cent ~ df.k.3.weak$base.tempo, data = df.k.3.weak)
summary(df.lm) #  *** F(125.2), p < 2.2e-16, centroid differences go down with tempo

# what about effects of tempo on centroid diff with fast k = 3 group only for none coupling? 
boxplot(df.k.3.none$dist.cent ~ df.k.3.none$base.tempo, col=c("white", "lightgray"), df.k.3.none)
df.lm <- lm(df.k.3.none$dist.cent ~ df.k.3.none$base.tempo, data = df.k.3.none)
summary(df.lm) #  *** F(153.1), p < 2.2e-16, centroid differences go down with tempo


# what about effects of tempo on dispersion for fast, k==3 group only for medium coupling?
df.k.3.medium <- subset(df.k.3, df.k.3$coupling == 'medium')
boxplot(df.k.3.medium$tap.disp ~ df.k.3.medium$base.tempo, col=c("white", "lightgray"), df.k.3.medium)
df.lm <- lm(df.k.3.none$dist.cent ~ df.k.3.none$base.tempo, data = df.k.3.none)
summary(df.lm) # *** F(153.1) p < 2.2e-16

df.k.3.strong <- subset(df.k.3, df.k.3$coupling == 'strong') # empty, no taps responses to strong stimuli were included in k==3 group 

#######################




###### linear regression, check assumptions for k=3 tap dispersion and centroid diff

# observation of centroid is independent of dispersion? i think so... centroid doesnt depend on dispersion
#e.g could have same centroid location but completely different dispersion 

# normality?
# check if difference bewteen centroids follows normal dist
hist(df.k.3$dist.cent)
# linearity?
cor(df.k.3$dist.cent, df.k.3$tap.disp) # roughly linear 
# Homoscedasticity... check later

#### RUN linear regression analysis 

# linear regression on centroid difference vs. mean tempo
df.lm <- lm(df.k.3$dist.cent ~ df.k.3$stim.mean.tempo, data = df.k.3)
summary(df.lm)

                          Estimate Std. Error   t value    Pr(>|t|)    
  (Intercept)             1.2112573  0.0401709   30.15   <2e-16 ***
  df.k.3$stim.mean.tempo -0.0078537  0.0004917  -15.97   <2e-16 ***

# linear regression on tap.disp vs. mean tempo
df.lm <- lm(df.k.3$tap.disp ~ df.k.3$stim.mean.tempo, data = df.k.3)
summary(df.lm)
                         Estimate   Std. Error  t value Pr(>|t|)    
(Intercept)             0.1499022  0.0214854   6.977 9.49e-12 ***
df.k.3$stim.mean.tempo -0.0003525  0.0002630  -1.340    0.181    
---


#### CHECK FOR LINEAR MIXED EFFECT, can tap mean iti be explained as a linear model of base tempo + random effects of dispersion  #####
lme.model = lmer(df.k.3$dist.cent ~ df.k.3$stim.mean.tempo + (1|df.k.3$subject) , data=df.k.3)


hist(df.k.3$[apply(df.k.3, 1, function( x ) any( x > 1.5 ) ), ])


head(df)


