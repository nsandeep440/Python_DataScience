#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:35:25 2018

@author: sandeepn
"""

import numpy as np
import pandas as pd

# =============================================================================
# =============================================================================
# # The Pandas Series Object
# =============================================================================
# =============================================================================

# =============================================================================
# A Pandas Series is a one-dimensional array of indexed data.
# =============================================================================

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
data['b']

# =============================================================================
# We can even use noncontiguous or nonsequential indices
# =============================================================================
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
data[3]

# =============================================================================
# Series as specialized dictionary
# =============================================================================
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population
population['California':'Illinois']

# =============================================================================
# =============================================================================
# # The Pandas DataFrame Object
# =============================================================================
# =============================================================================
# =============================================================================
# DataFrame as a generalized NumPy array
# =============================================================================
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}

area = pd.Series(area_dict)
area

states = pd.DataFrame({'population': population, 'area': area})
states
states.index
states.columns
states['area']

# =============================================================================
# Constructing DataFrame objects
# =============================================================================

#From a single Series object
pd.DataFrame(population, columns=['population'])

#From a list of dicts
data = [{'a': i, 'b': 2*i} for i in range(5)]
data
pd.DataFrame(data)
###### OR
pd.DataFrame( [{'a': 1, 'c':2}, {'b': 3, 'e': 2}] )

# =============================================================================
# From a NumPy structured array
# =============================================================================

## numpy structued array
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)
s_data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
s_data 
s_data['name'] = name
s_data['age'] = age
s_data['weight'] = weight 
print(s_data)

pd.DataFrame(s_data)

# =============================================================================
# The Pandas Index Object
# This Index object is an interesting structure in itself, 
# and it can be thought of either as an immutable array or as an ordered set
# 
# =============================================================================
ind = pd.Index([2, 3, 5, 7, 11])
ind
ind[1]
ind[::2]

ind[1] = 0  #Index does not support mutable operations

# =============================================================================
# Index as ordered set
# =============================================================================
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB # intersection
indA | indB # union
indA ^ indB # symmetric difference

# =============================================================================
# Data Indexing and Selection
# =============================================================================

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
# slicing by explicit index  - 1Dim
data['a':'c']

# slicing by implicit integer index 
data[0:2]

# =============================================================================
# Indexers: loc, iloc, and ix
# =============================================================================
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data

# explicit index when indexing 
data[1] # => 'a'
data[3] # => 'b'

# implicit index when slicing 
data[1:3]  # these are indexing 0th index, 1stindex, 2nd index and so on.
# output
#3    b
#5    c

# =============================================================================
# Because of this potential confusion in the case of integer indexes,
# Pandas provides some special indexer attributes that explicitly expose certain indexing schemes
# =============================================================================
# =============================================================================
# First, the loc attribute allows indexing and slicing that always references the explicit index
# =============================================================================
data.loc[1] # => 'a'
data.loc[3] # => 'b'
data.loc[1:3]  # these are explicit index -> index at key 1, index at key 3
# output -> compare from implicit index data[1:3]
#1    a
#3    b

# =============================================================================
# The iloc attribute allows indexing and slicing that always references the implicit
# Python-style index
# =============================================================================

data.iloc[1] # value at index 1.
data.iloc[1:3]
# output
#3    b
#5    c

# =============================================================================
# =============================================================================
# # Handling Missing Data
# =============================================================================
# =============================================================================

# =============================================================================
# None: Pythonic missing data
# =============================================================================
vals1 = np.array([1, None, 3, 4])
vals1 ## object
vals1.sum() ## error

# =============================================================================
# NaN: Missing numerical data
# =============================================================================

vals2 = np.array([1, np.nan, 3, 4])
vals2
print(vals2.dtype)
vals2.sum(), vals2.min(), vals2.max()
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

# =============================================================================
# NaN and None in Pandas
# =============================================================================
pd.Series([1, np.nan, 2, None])

# =============================================================================
# output
# 0    1.0
# 1    NaN
# 2    2.0
# 3    NaN
# =============================================================================

# =============================================================================
# Operating on Null Values
# =============================================================================
# =============================================================================
# isnull()
#   Generate a Boolean mask indicating missing values
# notnull()
#   Opposite of isnull()
# dropna()
#   Return a filtered version of the data
# fillna()
#   Return a copy of the data with missing values filled or imputed
# =============================================================================

# =============================================================================
# Detecting null values
# =============================================================================
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]

# =============================================================================
# Dropping null values
# =============================================================================
data.dropna()

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df

df.dropna() # rows are dropped
df.dropna(axis='columns')
df[3] = np.nan
df
df.dropna(axis='columns', how='all') # drop columns if all na's
df.dropna(axis='rows', thresh=3) # min numb of non null values to be kept
# =============================================================================
# Filling null values
# =============================================================================
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
# =============================================================================
# We can specify a forward-fill to propagate the previous value forward:
# =============================================================================
# forward-fill

data.fillna(method='ffill')

# =============================================================================
# =============================================================================

# =============================================================================
# we can specify a back-fill to propagate the next values backward:
# =============================================================================
# back-fill

data.fillna(method='bfill')

df.fillna(method='ffill', axis=1) # if prev value is not available it will remains same.

# =============================================================================
# =============================================================================


# =============================================================================
# Combining Datasets: Concat and Append
# =============================================================================
#     Recall: Concatenation of NumPy Arrays
# =============================================================================
x,y,z=[1,2,3],[4,5,6],[7,8,9]
np.concatenate([x, y, z])
# =============================================================================
# Simple Concatenation with pd.concat
# =============================================================================
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])

def make_df(cols, ind): 
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind] 
        for c in cols}
    return pd.DataFrame(data, ind) # example DataFrame
           
make_df('ABC', range(3))

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(pd.concat([df1,df2]))

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(pd.concat([df3,df4]))

# =============================================================================
# Duplicate indices
# =============================================================================
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index # make duplicate indices!
print(pd.concat([x, y]))

# =============================================================================
# Concatenation with joins
# =============================================================================
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(pd.concat([df5,df6], join='inner')) ## intersection - column wise
# outer => union
print(pd.concat([df5,df6], join_axes=[df5.columns]))
# =============================================================================

# =============================================================================
# =============================================================================
# # Combining Datasets: Merge and Join
# =============================================================================

# =============================================================================
# Categories of Joins
# one-to-one, many-to-one, and many-to-many joins.
# =============================================================================
# =============================================================================
# One-to-one joins

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]}) 
print(df1); print(df2)
df3 = pd.merge(df1, df2); df3
# =============================================================================

# =============================================================================
# Many-to-one joins

df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
df4
pd.merge(df3, df4)
# =============================================================================

# =============================================================================
# Many-to-many joins

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting','Engineering', 'Engineering', 'HR', 'HR'], 
    'skills': ['math', 'spreadsheets', 'coding', 'linux','spreadsheets', 'organization']})
df5
pd.merge(df1, df5)


# =============================================================================

# =============================================================================


























