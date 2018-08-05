# -*- coding: utf-8 -*-
"""
Created on 2018/07/29

@author: Xavi
"""

################################################################################
# 1. Análisis Preliminar y revisión de caracteristicas

# a) Load libraries
#Importing libraries
import pandas as pd #data manipulation
import os
import numpy as np # mathematical operations
import scipy as sci # math ops
import seaborn as sns # visualizations
import matplotlib.pyplot as plt # for plottings

#
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split

# To model the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
import re

# b) Data Analysis

reg_season.describe()
reg_season.head(5)

#Add the differential
reg_season['Diff'] = reg_season['WScore'] - reg_season['LScore'] 
reg_season['True_Outcome'] = np.where(reg_season['Diff'] > 5, 'Y','N')


#Massey data
#..........................................................................
massey.head(5)

#What are the unique rankings? -- Looks like every team gets rated 1-351
massey['SystemName'].sort_values().unique()


""" Create a table of average of all end of year ratings """

#Drop duplicates from each rating system keep only max rating day
ratings = massey.sort_values('RankingDayNum', ascending = False).drop_duplicates(['Season','SystemName','TeamID'])

#Get the mean rating for each season team combo across all rating systems
ratings2 = ratings.groupby(['Season', 'TeamID'], as_index = False)['OrdinalRank'].mean()

#Merge with the team name table to get the team name 
ratings3 = ratings2.merge(teams, on = 'TeamID')
ratings3.head(2)

# =============================================================================
#    Season  TeamID  OrdinalRank   TeamName  FirstD1Season  LastD1Season
# 0    2003    1102   154.058824  Air Force           1985          2018
# 1    2004    1102    46.513514  Air Force           1985          2018
# =============================================================================

ratings3.Season.unique()
#only have ratings from 2003 - 2017

#Add winning team rank, merging on team ID and Season
reg_season_ratings = reg_season[reg_season['Season']>2002] \
                     .merge(ratings3[['Season','TeamID','OrdinalRank','TeamName']], \
                     left_on = ['WTeamID','Season'], right_on = ['TeamID','Season'])

reg_season_ratings.head(2)

# =============================================================================
#    Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT  Diff  \
# 0    2003      10     1104      68     1328      62    N      0     6   
# 1    2003      18     1104      82     1106      56    H      0    26   
# 
#   True_Outcome  TeamID  OrdinalRank TeamName  
# 0            Y    1104    36.638889  Alabama  
# 1            Y    1104    36.638889  Alabama  
# =============================================================================

reg_season_ratings.rename(columns = {'OrdinalRank':'WTeamRank', 'TeamName':'WTeamName'}, inplace = True)

reg_season_ratings2 = reg_season_ratings.merge(ratings3[['Season','TeamID','OrdinalRank','TeamName']], \
                left_on = ['LTeamID','Season'], right_on = ['TeamID','Season'])

reg_season_ratings2.rename(columns = {'OrdinalRank':'LTeamRank', 'TeamName':'LTeamName'}, inplace = True)

reg_season_ratings2.head(2)

#Drop duplicate column
reg_season_ratings2.drop(['TeamID_x','TeamID_y'], 1, inplace = True)


#Preparamos datos para construir el modelo
#......................................................................................
""" Lets use only the both team's seed and the avg. rating to predict winner using Naive Bayes """
tourney_seeds.head(2)

#Step 1: Pull in the features (seeds and ratings)
tourney_results1 = tourney_results.merge(tourney_seeds, left_on = ['WTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results2 = tourney_results1.merge(tourney_seeds, left_on = ['LTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results3 = tourney_results2.merge(ratings3, left_on = ['WTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results4 = tourney_results3.merge(ratings3, left_on = ['LTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results5 = tourney_results4[['Season','WTeamID', 'LTeamID', 'TeamName_x', 'TeamName_y', \
                                    'Seed_x', 'Seed_y','OrdinalRank_x','OrdinalRank_y']] \
    .rename(columns = {'Seed_x':'WSeed','Seed_y': 'LSeed','TeamName_x' : 'WTeamName', \
                          'TeamName_y' : 'LTeamName', 'OrdinalRank_x' : 'WRank','OrdinalRank_y' : 'LRank'})

tourney_results5.head()

#selecting specific columns of tourney results
tourney_results6 = tourney_results5[['Season', 'WTeamID', 'LTeamID', 'WTeamName', 'LTeamName', \
                                    'WSeed', 'LSeed','WRank', 'LRank']]

#randomizing dataset
tourney_results6 = tourney_results6.sample(frac=1).reset_index(drop=True)

tourney_results6.head()


#splitting dataframe into two equal halves, so we can add multiple classifications for W/L
if (len(tourney_results6) % 2 > 0):
    print(len(tourney_results6))
    tourney_results6 = tourney_results6[:-1]
    print('trimming dataframe to even length so we can split')
    print(len(tourney_results6))
    
tourney_results7 = np.split(tourney_results6, 2)

