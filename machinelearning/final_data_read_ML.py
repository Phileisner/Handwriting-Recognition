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

def read_st_csvfile(stfilename):

    df = pandas.read_csv(stfilename, header = 4, usecols=[5,6,7])
    # print(df)
    dfnp = df.to_numpy()
    #print(dfnp)
    return dfnp

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

def readsensordata(listof3files,maxdatalength,makesamelength):
    
    sensordata = np.zeros((maxdatalength,9))
    
    for fname in listof3files:
        if 'Accel' in fname:
            
            c1 = 0
            
        elif 'Gyro' in fname:
            
            c1 = 3
            
        elif 'Magnet' in fname:
           
            c1 = 6
            
        else:
            
            print('ERROR:  Data filename does not contain ACCEL, GYRO, OR MAGNET.  Dataset may be invalid')
            c1 = 0

        
        filedata = read_st_csvfile(fname)
        
        npts = np.size(filedata,0)

        if makesamelength==1:
            
            tfile = np.arange(0,npts)/(npts-1)*maxdatalength
            t = np.arange(0,maxdatalength)
            newfiledata = np.zeros((maxdatalength,3))
            
            for k in range(3):
                newfiledata[:,k] = np.interp(t,tfile,filedata[:,k])
                nptsnew = maxdatalength
        else:
            newfiledata = filedata
            nptsnew = npts
            
        
        sensordata[0:nptsnew,c1:(c1+3)] = newfiledata
        
        
    return sensordata,npts

dirlist = glob.glob("C:/Users/Philip/Desktop/Organized_Data/*")

nptsv = []
paramset = np.zeros((1,48))
k = 0
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
        
maxdatalength = max(nptsv)
ntrials = np.size(nptsv,0)
print(' ')
print('Data size:  number of trials = ',ntrials,' and max data length in file = ',maxdatalength)


paramset_norm = copy.deepcopy(paramset)

normaccel = np.mean(paramset[:,14])
print('accelerometer normalization = ',normaccel)
paramset_norm[:,0:16] = paramset_norm[:,0:16]/normaccel

normgyro = np.mean(paramset[:,30])
paramset_norm[:,16:32] = paramset_norm[:,16:32]/normaccel
print('gyroscope normalization = ',normgyro)

normmagnet = np.mean(paramset[:,46])
print('magnetometer normalization = ',normmagnet)
paramset_norm[:,32:48] = paramset_norm[:,32:48]/normaccel

useaccelerometer = 1
usegyroscope = 0
usemagnetometer = 0
makesamelength = 1

totalcolumns = 3*(useaccelerometer+usegyroscope+usemagnetometer)*maxdatalength

dataset = np.zeros((ntrials,totalcolumns))

target = []

k = 0
for subdir in dirlist:
    
    digit = int(subdir[len(subdir)-1])
    subdirlist = glob.glob(subdir+"/*")
    
    subdirlist.sort()
    #print(subdirlist)
    for subsubdir in subdirlist:
        listoffiles = glob.glob(subsubdir+"/*.csv")
        listoffiles.sort()

        
        thisdata,npts = readsensordata(listoffiles,maxdatalength,makesamelength)
        
        #print((np.size(thisdata,0),np.size(thisdata,1)))
        
        datasetrow = np.zeros((1,totalcolumns))
        if useaccelerometer==1 and usegyroscope==1 and usemagnetometer==1:
            # accelerometer
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,0].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,1].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,2].transpose()
            # gyroscope
            datasetrow[3*maxdatalength:4*maxdatalength] = thisdata[0:maxdatalength,3].transpose()
            datasetrow[4*maxdatalength:5*maxdatalength] = thisdata[0:maxdatalength,4].transpose()
            datasetrow[5*maxdatalength:6*maxdatalength] = thisdata[0:maxdatalength,5].transpose()
            # magnetometer
            datasetrow[6*maxdatalength:7*maxdatalength] = thisdata[0:maxdatalength,6].transpose()
            datasetrow[7*maxdatalength:8*maxdatalength] = thisdata[0:maxdatalength,7].transpose()
            datasetrow[8*maxdatalength:9*maxdatalength] = thisdata[0:maxdatalength,8].transpose()
        elif useaccelerometer==1 and usegyroscope==1 and usemagnetometer==0:
            # accelerometer
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,0].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,1].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,2].transpose()
            # gyroscope
            datasetrow[3*maxdatalength:4*maxdatalength] = thisdata[0:maxdatalength,3].transpose()
            datasetrow[4*maxdatalength:5*maxdatalength] = thisdata[0:maxdatalength,4].transpose()
            datasetrow[5*maxdatalength:6*maxdatalength] = thisdata[0:maxdatalength,5].transpose()
        elif useaccelerometer==1 and usegyroscope==0 and usemagnetometer==1:
            # accelerometer
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,0].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,1].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,2].transpose()
            # magnetometer
            datasetrow[3*maxdatalength:4*maxdatalength] = thisdata[0:maxdatalength,6].transpose()
            datasetrow[4*maxdatalength:5*maxdatalength] = thisdata[0:maxdatalength,7].transpose()
            datasetrow[5*maxdatalength:6*maxdatalength] = thisdata[0:maxdatalength,8].transpose()
        elif useaccelerometer==1 and usegyroscope==0 and usemagnetometer==0:
            # accelerometer
            datasetrow[0,0:maxdatalength] = thisdata[0:maxdatalength,0].transpose()
            datasetrow[0,maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,1].transpose()
            datasetrow[0,2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,2].transpose()
        elif useaccelerometer==0 and usegyroscope==1 and usemagnetometer==1:
            # gyroscope
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,3].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,4].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,5].transpose()
            # magnetometer
            datasetrow[3*maxdatalength:4*maxdatalength] = thisdata[0:maxdatalength,6].transpose()
            datasetrow[4*maxdatalength:5*maxdatalength] = thisdata[0:maxdatalength,7].transpose()
            datasetrow[5*maxdatalength:6*maxdatalength] = thisdata[0:maxdatalength,8].transpose()
        elif useaccelerometer==0 and usegyroscope==1 and usemagnetometer==0:
            # gyroscope
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,3].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,4].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,5].transpose()
        elif useaccelerometer==0 and usegyroscope==0 and usemagnetometer==1:
            # magnetometer
            datasetrow[0:maxdatalength] = thisdata[0:maxdatalength,6].transpose()
            datasetrow[maxdatalength:2*maxdatalength] = thisdata[0:maxdatalength,7].transpose()
            datasetrow[2*maxdatalength:3*maxdatalength] = thisdata[0:maxdatalength,8].transpose()
        else:
            print('NO DATA SELECTED!  Dataset creation will fail.')
        
        dataset[k,:] = datasetrow
        target.append(digit)

        k = k + 1
        
X_train, X_test, y_train, y_test = train_test_split(paramset_norm,target,test_size=0.1,stratify=target,random_state=8)
# this does the training of the neural net
mlp = MLPClassifier(solver='lbfgs',random_state=0,max_iter=1500,hidden_layer_sizes=(50,25)).fit(X_train,y_train)
# output the results
answer = mlp.predict(X_test)
print('Test set results')
print('Predicted digit   Actual digit')
for i in range(len(y_test)):
    print('             ',answer[i],'   ',y_test[i])
print(' ')
print('Accuracy on training set: {:.2f}'.format(mlp.score(X_train,y_train)))
print('Accuracy on test set: {:.2f}'.format(mlp.score(X_test,y_test)))

#normchoice = 0