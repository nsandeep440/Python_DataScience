#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:44:01 2018

@author: sandeepn
"""

import numpy as np

# =============================================================================
# Understanding Data Types in Python
# In Python the types are dynamically inferred
# 
# =============================================================================

result = 0
for i in range(100):
    result += i
    
x = 4    
x
x = "Four"
x

# =============================================================================
# Python List
# List of integers
# =============================================================================

L = list(range(10))
L
type(L) 
type(L[0])

# list of strings
l2 = [str(c) for c in L]
l2
type(l2[0])

# =============================================================================
# Because of Python’s dynamic typing, we can even create heterogeneous lists
# =============================================================================

l3 = [True, 1, 0.3, "one"]
type(l3[0])
[type(index) for index in l3]


# =============================================================================
# Fixed-Type Arrays in Python
# Python offers several different options for storing data in efficient, fixed-type data buffers. 
# The built-in array module can be used to create dense arrays of a uniform type
# =============================================================================

import array
A = array.array('i', L)
A
# =============================================================================
# output: array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Here 'i' is a type code indicating the contents are integers.
# =============================================================================

# =============================================================================
# Creating Arrays from Python Lists
# =============================================================================
np.array([1, 4, 2, 5, 3])
# =============================================================================
# NumPy is constrained to arrays that all contain the same type. 
# If types do not match, NumPy will upcast if possible
# =============================================================================
np.array([3.14, 4, 2, 3])

#explicitly set the data type of the resulting array
np.array([1, 4, 2, 5, 3], dtype="float32")

# =============================================================================
# unlike Python lists, NumPy arrays can explicitly be multidimensional
# =============================================================================

np.array([range(i, i+3) for i in [2,9,3]])

# =============================================================================
# Creating Arrays from Scratch
# =============================================================================

np.zeros(10, dtype=int)

np.ones((2,4), dtype=float)

np.full((3,5), 3.14) #, dtype=int)

np.arange(0, 10, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(1,2, num=5)

# Create a 3x3 array of uniformly distributed 
# random values between 0 and 1
np.random.random(size=(3,3))

# =============================================================================
# Create a 3x3 array of normally distributed random values 
# with mean 0 and standard deviation 1
# =============================================================================
np.random.normal(0, 1, size=(3,3))

# =============================================================================
# Create a 3x3 array of random integers in the interval [0, 10)
# =============================================================================
np.random.randint(0, high=10, size=(3,4))

# =============================================================================
# Create a 3x3 identity matrix
# =============================================================================
np.eye(4)

# =============================================================================
# Attributes of arrays
    # Determining the size, shape, memory consumption, and data types of arrays
# Indexing of arrays
    # Getting and setting the value of individual array elements
# Slicing of arrays
    # Getting and setting smaller subarrays within a larger array
# Reshaping of arrays
    # Changing the shape of a given array
# Joining and splitting of arrays
    # Combining multiple arrays into one, and splitting one array into many
# =============================================================================

# =============================================================================
# NumPy Array Attributes
# =============================================================================

np.random.seed(0) # seed for reproducibility
x1 = np.random.randint(10, size=6) # One-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array

print("x3 ndim: ", x3.ndim) 
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


# =============================================================================
# =============================================================================
# # Array Indexing: Accessing Single Elements
# =============================================================================
# =============================================================================

x1
x1[0]
x1[5]

# =============================================================================
# To index from the end of the array, you can use negative indices:
# =============================================================================

x1[-1] ## => x1[5]
x1[-6]

# =============================================================================
# In a multidimensional array, you access items using a comma-separated tuple of indices
# =============================================================================
x2

x2[0,3]
x2[0, -1]
x2[2,0]
x2[2,-4]

# =============================================================================
# modify values using any of the above index notation:
# =============================================================================
x2[2,0] = 14
x2
x2[2,-4] = 13

# =============================================================================
# =============================================================================
# # Array Slicing: Accessing Subarrays
# # x[start:stop:step]
# =============================================================================
# =============================================================================

#One-dimensional subarrays

x = np.arange(10)
x

x[:5] # first five elements 0r x[0:5]
x[5:] # elements after index 5
x[4:7] # middle subarray
x[::2] # every other element
x[1::2] # every other element, starting at index 1

# =============================================================================
# A potentially confusing case is when the step value is negative. 
# In this case, the defaults for start and stop are swapped. 
# This becomes a convenient way to reverse an array:
# =============================================================================
x[::-1] # all elements, reversed
x[5::-3] # reversed every other from index 5

# =============================================================================
# Multidimensional subarrays
# multiple slices separated by com‐ mas
# =============================================================================

x2[:2, :3] # two rows, three columns
x2[:3, :2]
x2[:3, ::2] # all rows, every other column

# =============================================================================
# Accessing array rows and columns.
# =============================================================================

print(x2[:, 0]) # first column of x2
print(x2[1, :]) # second row of x2
print(x2[1:, :]) #from second row with all columns

# =============================================================================
# Subarrays as no-copy views
# =============================================================================
x2_sub = x2[:2, :2] 
print(x2_sub)

# =============================================================================
# Reshaping of Arrays
# =============================================================================

x = np.array([1, 2, 3])
x.reshape((1,3)) # row vector via reshape
x[np.newaxis, :]  # row vector via newaxis


np.random.seed(0)
def compute_reciprocals(values): 
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i] 
    return output

big_array = np.random.randint(1, 100, size=1000000)
big_array
%timeit compute_reciprocals(big_array)

# =============================================================================
# =============================================================================
# # Introducing UFuncs
# # This is known as a vectorized operation.
# # Vectorized operations in NumPy are implemented via ufuncs
# =============================================================================
# =============================================================================

%timeit (1.0 / big_array)

# =============================================================================
# Another excellent source for more specialized and obscure ufuncs is the submodule scipy.special
# =============================================================================

from scipy import special
# Gamma functions (generalized factorials) and related functions x=[1,5,10]
print("gamma(x) =", special.gamma(x)) 
print("ln|gamma(x)| =", special.gammaln(x)) 
print("beta(x, 2) =", special.beta(x, 2))

# =============================================================================
# Advanced Ufunc Features
# =============================================================================

# =============================================================================
# Specifying output
# =============================================================================

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y) 
print(y)

y = np.zeros(10)
np.power(3, x, out=y[::2])
print(y)

# =============================================================================
# Aggregates
# =============================================================================
x = np.arange(1, 6)
x
np.add.reduce(x)
np.multiply.reduce(x)
np.add.accumulate(x)
np.multiply.accumulate(x)


X = np.random.random((10, 3))
Xmean = X.mean(0)
X_centered = X - Xmean
X_centered.mean(0)

# =============================================================================
# =============================================================================
# # Fancy Indexing
# =============================================================================
# =============================================================================
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)

[x[3], x[7], x[2]]

ind = [3, 7, 4]
x[ind]

ind = np.array([[3, 7], [4, 5]])
x[ind]

# =============================================================================
# =============================================================================
# # Structured Data: NumPy’s Structured Arrays
# =============================================================================
# =============================================================================

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
data 
data['name'] = name
data['age'] = age
data['weight'] = weight 
print(data)

data['name']
data[0]['name']

data[data['age'] < 30]['name']
