setwd('/Users/nolanlem/Desktop/takako-2-10-23/final-rev/')
######### SEX  (all reported)######

#create table
data <- matrix(c(6, 9, 3, 11, 15, 6), ncol=3, byrow=TRUE)
colnames(data) <- c("Reg","Hyb","Fast")
rownames(data) <- c("Female","Male")
sdata <- as.table(data)

#view table
sdata

#         Reg Hyb Fast
# Female   6  9    3
# Male    11  15    6

#Perform Chi-Square Test of Independence
chisq.test(sdata)
# Pearson's Chi-squared test
# 
# data:  sdata
# X-squared = 0.33456, df = 2, p-value = 0.846

data:  sdata
X-squared = 0.054892, df = 2, p-value = 0.9729

# Since the p-value of the test is not less than 0.05, 
# we fail to reject the null hypothesis. 
# This means we do not have sufficient evidence to say that 
# there is an association between sex and group affiliation.
# In other words, sex and grouping are independent.

######### HANDEDNESS (all reported) #######

data <- matrix(c(15, 16, 8, 2, 8, 1), ncol=3, byrow=TRUE)
colnames(data) <- c("Reg","Hyb","Fast")
rownames(data) <- c("Right","Left")
hdata <- as.table(data)
#view table
hdata

 
#       Reg Hyb Fast
# Right  15  18    8
# Left    2   8    1

#Perform Chi-Square Test of Independence
chisq.test(hdata)

# Pearson's Chi-squared test
# 
# data:  hdata
# X-squared = 2.884, df = 2, p-value = 0.2365
data:  hdata
X-squared = 3.4561, df = 2, p-value = 0.1776

# Handness also is independent of group affiliation

####################### music
#setwd('/Users/tfujioka/Documents/Nolan/')
mdata <- read.csv("myrs_kura.csv", header = T)

# check normality
shapiro.test(mdata$myrs)
# Shapiro-Wilk normality test
# 
# data:  mdata$myrs
# W = 0.73319, p-value = 1.722e-07

W = 0.72311, p-value = 0.0000001855

# this means data are not normally distributed hence you can't do ANOVA

# instead, run Kruskal Wallis test
kruskal.test(myrs ~ group, data = mdata)

# Kruskal-Wallis rank sum test
# 
# data:  myrs by group
# Kruskal-Wallis chi-squared = 3.2007, df = 2, p-value = 0.2018
Kruskal-Wallis chi-squared = 2.4049, df = 2, p-value = 0.3005

# so groups are not different

##################################### age

adata <- read.csv("age_kura.csv", header = T)

# check normality
shapiro.test(adata$age)
# Shapiro-Wilk normality test
# 
# data:  adata$age
# W = 0.91084, p-value = 0.0008731

W = 0.90201, p-value = 0.0005602


# this means data are not normally distributed hence you can't do ANOVA

# instead, run Kruskal Wallis test
kruskal.test(age ~ group, data = adata)

# Kruskal-Wallis rank sum test
# 
# data:  age by group
# Kruskal-Wallis chi-squared = 0.37784, df = 2, p-value = 0.8279

Kruskal-Wallis chi-squared = 0.38819, df = 2, p-value = 0.8236


adata.reg <- subset(adata, group == 'reg')
adata.hyb <- subset(adata, group == 'hyb')
adata.fast <- subset(adata, group == 'fast')

mean(adata.reg$age)
mean(adata.hyb$age)
mean(adata.fast$age)

sd(adata.reg$age)
sd(adata.hyb$age)
sd(adata.fast$age)

# > with(adata, tapply(age, list(group), mean))
# fast      hyb      reg 
# 34.11111 36.57692 37.23529 
# > with(adata, tapply(age, list(group), sd))
# fast      hyb      reg 
# 10.14205 11.53490 12.56249 