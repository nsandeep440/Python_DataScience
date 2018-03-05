#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:07:25 2018

@author: sandeepn
"""

import os
import pandas as pd
import numpy as py

path = '/Users/sandeepn/Desktop/INSOFE/Course Files/Python_San/Python_DataScience/Python_DataScience/Intro_to_numpy_Pandas/Merge_Join_Example'
os.chdir(path)

os.listdir(path)

# =============================================================================
# Read csv files
# =============================================================================
population = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbrevs = pd.read_csv('state-abbrevs.csv')

print(population.head())
areas.head()
abbrevs.head()

# =============================================================================
# merge
# =============================================================================

merged = pd.merge(population, abbrevs, how='outer', left_on='state/region', right_on = 'abbreviation')
merged = merged.drop('abbreviation', 1)
merged.head()

# =============================================================================
# Check for any mismatches
# =============================================================================
merged.isnull() ### For each row

# =============================================================================
# check null for each column
# =============================================================================
merged.isnull().any()

## population and state have null values

merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(), 'state/region'].unique()

merged.loc[merged['state/region'] == "PR", 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == "USA", 'state'] = 'United States'
merged.isnull().any()

final = pd.merge(merged, areas, on = 'state', how = 'left')
final.head()
final.isnull().any()
# =============================================================================
# get the state with null area
# =============================================================================
final['state'][final['area (sq. mi)'].isnull()].unique()

final.dropna(inplace=True)
final.head()
final.isnull().any()

# =============================================================================
# select the portion of the data corresponding with the year 2000, and the total population
# =============================================================================

# =============================================================================
# We’ll use the query() function
# =============================================================================
# this requires the "numexpr" package
# =============================================================================
# =============================================================================

data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
# =============================================================================
# let’s compute the population density and display it in order
# We’ll start by rein‐ dexing our data on the state, and then compute the result
# =============================================================================
data2010.set_index('state', inplace=True) #### changing the column state to zero-index
desity = data2010['population'] / data2010['area (sq. mi)']


























