# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:48:49 2022

@author: nespi
"""
print("DATA-51100, Spring 2022")
print("NAME: Nick Spirakis")
print("PROGRAMMING ASSIGNMENT #3")

import numpy as np


#load and parse training data in a NumPy ndarray

#assume exactly 4 attribute values in the training and testing examples



test_file_name = "iris-testing-data.csv"
test_raw_data = open(test_file_name, 'rb')
test_data = np.genfromtxt(test_file_name, dtype=str, delimiter=',')


test_float_data = []
test_string_data = []

#-----------------------------------------------------------------------------
sepalLengthX = []
sepalWidthX = []
petalLengthX = []
petalWidthX = []
classKeeperX = []

ncount = 1
for n in test_data:
    one = []
    two = []
    three = []
    four = []
    five = []
    
    for i in n:
        if ncount == 1:
            one.append(i)
            ncount += 1
            
        elif ncount == 2:
            two.append(i)
            ncount += 1
            
        elif ncount == 3:
            three.append(i)
            ncount += 1
            
        elif ncount == 4:
            four.append(i)
            ncount += 1
        
        elif ncount == 5:
            classKeeperX.append(i)
            
    sepalLengthX.append(one)
    sepalWidthX.append(two)
    petalLengthX.append(three)
    petalWidthX.append(four)
    ncount = 1

slX = np.array(sepalLengthX)
flo_slX = slX.astype(np.float) #turns tfd into floats

swX = np.array(sepalWidthX)
flo_swX = swX.astype(np.float) #turns tfd into floats

plX = np.array(petalLengthX)
flo_plX = plX.astype(np.float) #turns tfd into floats

pwX = np.array(petalWidthX)
flo_pwX = pwX.astype(np.float) #turns tfd into floats

#-----------------------------------------------------------------------------

count = 0
#for loop
for n in test_data:
    a = []
    for i in n:
        if count < 4:
            a.append(i)
            #print(a)
            count += 1
        else:
            test_string_data.append(i)
    test_float_data.append(a)
    count = 0
            

tfd = np.array(test_float_data)
flo_tfd = tfd.astype(np.float) #turns tfd into floats
tsd = np.array(test_string_data)

#------------------------------------------------------------------------------

train_file_name = "iris-training-data.csv"
train_raw_data = open(train_file_name, 'rb')
train_data = np.genfromtxt(train_file_name, dtype=str, delimiter=',')


train_float_data = []
train_string_data = []


sepalLengthY = []
sepalWidthY = []
petalLengthY = []
petalWidthY = []
classKeeperY = []


mcount = 1
for n in train_data:
    oneY = []
    twoY = []
    threeY = []
    fourY = []
    fiveY = []
    
    for i in n:
        if mcount == 1:
            oneY.append(i)
            mcount += 1
            
        elif mcount == 2:
            twoY.append(i)
            mcount += 1
            
        elif mcount == 3:
            threeY.append(i)
            mcount += 1
            
        elif mcount == 4:
            fourY.append(i)
            mcount += 1
        
        elif mcount == 5:
            classKeeperY.append(i)
            
    sepalLengthY.append(oneY)
    sepalWidthY.append(twoY)
    petalLengthY.append(threeY)
    petalWidthY.append(fourY)
    mcount = 1




slY = np.array(sepalLengthY)
flo_slY = slY.astype(np.float) #turns tfd into floats

swY = np.array(sepalWidthY)
flo_swY = swY.astype(np.float) #turns tfd into floats

plY = np.array(petalLengthY)
flo_plY = plY.astype(np.float) #turns tfd into floats

pwY = np.array(petalWidthY)
flo_pwY = pwY.astype(np.float) #turns tfd into floats


#-----------------------------------------------------------------------------
count2 = 0
#for loop
for n in train_data:
    a2 = []
    for i in n:
        if count2 < 4:
            a2.append(i)
            count2 += 1
        else:
            train_string_data.append(i)
    train_float_data.append(a2)
    count2 = 0


trfd = np.array(train_float_data)
flo_trfd = trfd.astype(np.float) #turns tfd into floats
trsd = np.array(train_string_data)



#-----------------------------------------------------------------------------

#classifies each testing example

#output the true and predicted class label to the screen 
#AND save it into a NEW 1D array of strings.

#TO DO THIS: 
#first compute the DISTANCE VALUE for each pair of 
#training and tetsing example (their attribute values).

#THEN: for each test example, find the training example with the 
#closest distance (USE NumPy's VECTORIZED FUNCTIONS).

#distance value for each pair of training and testing ex attribute vals

calc_dist = np.sqrt((((flo_tfd)[:,np.newaxis] - (flo_trfd)[np.newaxis,:])**2).sum(2)).argmin(1)


#-----------------------------------------------------------------------------

yClassKeeper = np.array(classKeeperY)
xClassKeeper = np.array(classKeeperX)
predicted = yClassKeeper[calc_dist] 

print_count = 0

print('#, True, Predicted')
for _ in range(len(predicted)):
    print(print_count +1, xClassKeeper[print_count], predicted[print_count])
    print_count += 1
    


#-----------------------------------------------------------------------------


#compute the accuracy
#TO DO THIS: go through the array of class labels for testing examples and 
#compare the label stores in the array created in step 2

#THEN: count how many matches you get.

#THEN: output the number of matches, 
#divided by the number of testing examples as a percentage.

trues = (xClassKeeper == predicted) 
acc = (trues == True).sum() / trues.shape

print("Accuracy: {:.2f}%".format(100 * acc[0]))

