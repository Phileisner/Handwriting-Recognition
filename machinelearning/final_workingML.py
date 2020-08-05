# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:02:55 2020

@author: Philip
"""


import glob
import numpy as np
import pandas
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Reads a csv into dataframe, converts to a numpy array
def read_st_csvfile(stfilename):

    df = pandas.read_csv(stfilename, header = 4, usecols=[5,6,7])
    # print(df)
    dfnp = df.to_numpy()
    #print(dfnp)
    return dfnp

#creates array
def readsensordata_params(listof3files):
    nparams = (4*3 + 4)*3
    paramdata = np.zeros((1,nparams))
    
    for fname in listof3files:
        if 'Accel' in fname:
            
            c1 = 0
           
        elif 'Gyro' in fname:
            
            c1 = 16
            
        elif 'Magnet' in fname:
            
            c1 = 32
            
        else:
            
            print('ERROR:  Data filename does not contain ACCEL, GYRO, OR MAGNET.  Dataset may be invalid')
            c1 = 0

        filedata = read_st_csvfile(fname)
        
        npts = np.size(filedata,0)
        
        mins = np.min(filedata,0)
        maxs = np.max(filedata,0)
        means = np.mean(filedata,0)
        stdevs = np.std(filedata,0)
        magnitude = np.sqrt(np.power(filedata[:,0],2) + np.power(filedata[:,1],2) + np.power(filedata[:,2],2))
        
        minmag = np.min(magnitude,0)
        maxmag = np.max(magnitude,0)
        meanmag = np.mean(magnitude,0)
        stdevmag = np.std(magnitude,0)
        
        paramdata[0,c1+0:(c1+3)] = mins
        paramdata[0,c1+3:(c1+6)] = maxs
        paramdata[0,c1+6:(c1+9)] = means
        paramdata[0,c1+9:(c1+12)] = stdevs
        paramdata[0,c1+12] = minmag
        paramdata[0,c1+13] = maxmag
        paramdata[0,c1+14] = meanmag
        paramdata[0,c1+15] = stdevmag
    
    return paramdata,npts

#makes dataset for the NN
def makedataset(path):
    dirlist = glob.glob(path + "/*")

    nptsv = []
    paramset = np.zeros((1,48))
    k = 0
    target = []
    for subdir in dirlist:
        
        digit = int(subdir[len(subdir)-1])
        subdirlist = glob.glob(subdir+"/*")
        
        subdirlist.sort()
        
        for subsubdir in subdirlist:
            listoffiles = glob.glob(subsubdir+"/*.csv")
            listoffiles.sort()
    
            paramsetrow,npts = readsensordata_params(listoffiles)
            
            nptsv.append(npts)
            
            if k==0:
                paramset[0,:] = paramsetrow
            else:
                
                paramset = np.append(paramset,paramsetrow,axis=0)
                
            k = k + 1
            target.append(digit)
            
    maxdatalength = max(nptsv)
    ntrials = np.size(nptsv,0)
    print(' ')
    print('Data size:  number of trials = ',ntrials,' and max data length in file = ',maxdatalength)
    
    
    
    
    return paramset,target

trainArray,targetArray = makedataset("C:/Users/Philip/Desktop/Organized_Data")

#normalization
paramset_norm = copy.deepcopy(trainArray)
normaccel = np.mean(paramset_norm[:,14])
print('accelerometer normalization = ',normaccel)
paramset_norm[:,0:16] = paramset_norm[:,0:16]/normaccel

normgyro = np.mean(paramset_norm[:,30])
paramset_norm[:,16:32] = paramset_norm[:,16:32]/normgyro
print('gyroscope normalization = ',normgyro)

normmagnet = np.mean(paramset_norm[:,46])
print('magnetometer normalization = ',normmagnet)
paramset_norm[:,32:48] = paramset_norm[:,32:48]/normmagnet

#training and testing NN        
X_train, X_test, y_train, y_test = train_test_split(paramset_norm,targetArray,test_size=0.1,stratify=targetArray,random_state=25)
# this does the training of the neural net
mlp = MLPClassifier(solver='lbfgs',random_state=0,max_iter=1500,hidden_layer_sizes=(24,12)).fit(X_train,y_train)
# output the results
answer = mlp.predict(X_test)
print('Test set results')
print('Predicted digit   Actual digit')
for i in range(len(y_test)):
    print('             ',answer[i],'   ',y_test[i])
print(' ')
print('Accuracy on training set: {:.2f}'.format(mlp.score(X_train,y_train)))
print('Accuracy on test set: {:.2f}'.format(mlp.score(X_test,y_test)))


#Allows for new data to test the model
newPath = input("Enter file path for new test")

testSet, targetSet = makedataset(newPath)

paramset_norm2 = copy.deepcopy(testSet)

print('accelerometer normalization = ',normaccel)
paramset_norm2[:,0:16] = paramset_norm2[:,0:16]/normaccel


paramset_norm2[:,16:32] = paramset_norm2[:,16:32]/normgyro
print('gyroscope normalization = ',normgyro)


print('magnetometer normalization = ',normmagnet)
paramset_norm2[:,32:48] = paramset_norm2[:,32:48]/normmagnet

answer2 = mlp.predict(paramset_norm2)
print('Test set results')
print('Predicted digit   Actual digit')
for i in range(len(targetSet)):
    print('             ',answer2[i],'   ',targetSet[i])
print('Accuracy on test set: {:.2f}'.format(mlp.score(paramset_norm2,targetSet)))

