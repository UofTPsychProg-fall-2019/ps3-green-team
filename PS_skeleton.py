#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pandorable problem set 3 for PSY 1210 - Fall 2019

@author: katherineduncan

In this problem set, you'll practice your new pandas data management skills,
continuing to work with the 2018 IAT data used in class

Note that this is a group assignment. Please work in groups of ~4. You can divvy
up the questions between you, or better yet, work together on the questions to
overcome potential hurdles
"""

#%% import packages
import os
import numpy as np
import pandas as pd

#%%
# Question 1: reading and cleaning

# read in the included IAT_2018.csv file
path = os.getcwd()
data_file = path + '/IAT/IAT_2018.csv'
IAT = pd.read_csv(data_file)

# rename and reorder the variables to the following (original name->new name):
# session_id->id
# genderidentity->gender
# raceomb_002->race
# edu->edu
# politicalid_7->politic
# STATE -> state
# att_7->attitude
# tblacks_0to10-> tblack
# twhites_0to10-> twhite
# labels->labels
# D_biep.White_Good_all->D_white_bias
# Mn_RT_all_3467->rt

IAT = IAT.rename(columns={'session_id':'id', 'genderidentity':'gender',
'raceomb_002':'race', 'politicalid_7':'politic', 'STATE': 'state', 'att_7':'attitude',
'tblacks_0to10':'tblack','twhites_0to10':'twhite','D_biep.White_Good_all':'D_white_bias',
'Mn_RT_all_3467':'rt'}) #renames previous labels (pre-colon) to what is specified
#after the colon.

# remove all participants that have at least one missing value
IAT_clean = IAT.dropna(how='any', axis=0)


# check out the replace method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
# use this to recode gender so that 1=men and 2=women (instead of '[1]' and '[2]')
IAT_clean = IAT_clean.replace(to_replace={'[1]', '[2]'}, value={1, 2})

# use this cleaned dataframe to answer the following questions

#%%
# Question 2: sorting and indexing --> Rebekah

# use sorting and indexing to print out the following information:

# the ids of the 5 participants with the fastest reaction times
IAT_by_rt = IAT_clean.sort_values(by=['rt']) #sort by rt
print(list(IAT_by_rt.iloc[0:5,0])) #print the corresponding ids (cast as list to suppress index)

# the ids of the 5 men with the strongest white-good bias
IAT_by_white_good = IAT_clean.sort_values(by=['D_white_bias'])
IAT_men = IAT_by_white_good[IAT_by_white_good['gender']==1]
print('men: ' + str(list(IAT_men.iloc[0:5,0])))

# the ids of the 5 women in new york with the strongest white-good bias
IAT_women = IAT_by_white_good[IAT_by_white_good['gender']==2]
IAT_women_NY = IAT_by_white_good[IAT_by_white_good['state']=='NY']
print('women in NY: ' + str(list(IAT_women_NY.iloc[0:5,0])))


#%%
# Question 3: loops and pivots --> Emily

# check out the unique method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html
# use it to get a list of states
states = list(IAT_clean.loc[:,'state'].unique()) #I'm getting 62 states; am I going crazy???
#Juliana: I'm also getting 62... I'm assuming the list also includes territories and islands
#(e.g., Guam).

# write a loop that iterates over states to calculate the median white-good
# bias per state
# store the results in a dataframe with 2 columns: state & bias
df = pd.DataFrame(columns=['state', 'bias'])
count = 0
for st in states:
    tmpMean = IAT_clean[IAT_clean['state']==st].median()[3] # Gaeun: this question seemed to require median, so 'mean -> median'
    df = df.append(pd.Series([st,tmpMean], index=df.columns), ignore_index=True)

# now use the pivot_table function to calculate the same statistics
state_bias= pd.pivot_table(IAT_clean, 'D_white_bias','state', aggfunc = [np.median])

# make another pivot_table that calculates median bias per state, separately
# for each race (organized by columns)
state_race_bias = pd.pivot_table(IAT_clean, 'D_white_bias','state','race', aggfunc = [np.median])

#%%
# Question 4: merging and more merging --> Juliana + Gaeun

# add a new variable that codes for whether or not a participant identifies as
# black/African American
IAT_clean['is_black']=1*(IAT_clean.race==5) # J - used an integer rather than a boolean
#so it would be easier for subsequent tasks.

# use your new variable along with the crosstab function to calculate the
# proportion of each state's population that is black
# *hint check out the normalization options
prop_black = pd.crosstab(IAT_clean.state, IAT_clean.is_black, normalize='index')
#Juliana - I chose to normalize by column, in order to give a proportion of non-black
#(0), to black (1) people within each state. Other normalization options did not
#yield this. Additionally, it made more sense to list .state as the index, and .is_black as the columns. 

# state_pop.xlsx contains census data from 2000 taken from http://www.censusscope.org/us/rank_race_blackafricanamerican.html
# the last column contains the proportion of residents who identify as
# black/African American
# read in this file and merge its contents with your prop_black table
census_file = path+'/state_pop.xlsx'
# pip install xlrd #<< Gaeun: for somebody doesn't have xlrd package yet:)
census = pd.read_excel(census_file) #use read_excel for .xlsx files
prop_black=prop_black.reset_index(level=['state']) #had to make state a column,
#as it was the index in the crosstab.
merged = pd.merge(census,prop_black,left_on='State', right_on='state')
#Juliana - Previously, I renamed the 'prop_black' 'state' to 'State' to merge. 
#Saw Gaeun's comment below and used the two parameters at the end instead.

# use the corr method to correlate the census proportions to the sample proportions
correlation = np.corrcoef(merged.per_black, merged.iloc[:,5]) # Gaeun: The last column(6th) was sample proprtion. 
# Correlation coefficient ~ .88

# now merge the census data with your state_race_bias pivot table
srB_reindex = state_race_bias.reset_index(level=['state'])
merged2 = pd.merge(census, state_race_bias, left_on = 'State', right_on = 'state') 
# Gaeun: when variable names for merging were different, we can try left/right_on as well.

# use the corr method again to determine whether white_good biases is correlated
# with the proportion of the population which is black across states
# calculate and print this correlation for white and black participants
col_forWhite = np.corrcoef(merged2.per_black, merged2.iloc[:,9])
col_forBlack = np.corrcoef(merged2.per_black, merged2.iloc[:,8])
print('correlation(white): ', col_forWhite[0,1], '\n' 
      'correlation(black): ', col_forBlack[0,1])
