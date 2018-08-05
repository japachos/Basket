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

# b) Load dataset

teams = pd.read_csv(r"C:\Users\Chuseta\Documents\Xavi\Big Data\Challenge\SportsAnalytics\Datasets\Stage1\DataFiles\Teams.csv")
reg_season = pd.read_csv(r"C:\Users\Chuseta\Documents\Xavi\Big Data\Challenge\SportsAnalytics\Datasets\Stage1\DataFiles\RegularSeasonCompactResults.csv") 

# Revisar datos leidos
teams.head()
reg_season.head()

