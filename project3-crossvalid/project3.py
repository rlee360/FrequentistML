# CODE REVIEW THURS 7/21

import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random

random.seed(0)
np.random.seed(0)

# textbook's Gaussian random vectors

num_data = 50
p = 5000

X = np.random.randn(num_data, p)   
#Y = np.random.randint(low=0, high=2, size=num_data)  # 50% 0's, 50% 1's
Y = np.array(int(num_data/2)*[0] + int(num_data/2)*[1]) # check le messenger

# train_test_split we will need to use kfold instead
#train_test_split(dataX, dataY, test_size=(valid_ratio+test_ratio), random_state=rand_seed)


#---------------------------------------------
# Parameters for K-nearest neighbors

k = 1  # how many nearest neighbors to use in the k-nearest neighbors


#---------------------------------------------
# correct way of doing cross-validation
#---------------------------------------------

threshold = 0.1  # threshold used to determine which predictors will be left in the model 
# The threshold is how correlated no? So we find the correlation greater than threshold?
# if so just do a slice of the rho array oh wait hmm - sort and slice = oh wait if we sort don't we lose the indices?
# We'd have to sort a copy, select the last 10 percent and then find those in the og array. Yikes
# aha
# Side note, I found another tool that we can perhaps try for better erm integration across several computers
# while not pretty, I'd still like to take a look just to see what it's like
# I am tempted to jump to a full blown ide like pycharm which has variable explorers and everything

# maybe should use top ten percent of predictors instead of thresholding
# so don't have to keep changing threshold

num_folds = 10
percent = np.zeros(num_folds)
num_test = int(num_data/num_folds)  # one_fold = num_test
num_train = num_data - num_test

# if we're going to call what comes next cross-validation,
# we should randomly shuffle the rows of both X and Y (in unison!) before breaking in folds
# (even though we know an extra shuffling won't really change anything because they're so random already)

perm = np.random.permutation(num_data)
X = X[perm]
Y = Y[perm]

'''
For reference:
In [7]: d                                                                       
Out[7]: 
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])

In [8]: d[::,2]                                                                 
Out[8]: array([ 2, 12, 22])
'''

for i in np.arange(0,num_folds):
    # randomly permutate before splitting into 10 ?
    test_row_indices = list(range(i*num_test,(i+1)*num_test)) #np.arange(i*num_test,(i+1)*num_test)  # the rows that for the hold-out set
    #test_row_indices = np.array([int(i) for i in test_row_indices])
    testX = X[test_row_indices]  # [ ,::]
    testY = Y[test_row_indices]
    
    # left_in_row_indices = np.arange(0,i*one_fold-1)
    trainX = np.delete(X,test_row_indices,axis=0)  # leave the hold-out set in the training set - expensive-sh? operation though...
    trainY = np.delete(Y,test_row_indices,axis=0)

    # testing for univariate correlation

    rho = np.zeros(p)

    for j in np.arange(0,p):
        x = trainX[::,j]  # a column consisting of a single predictor
        R = np.corrcoef(x,trainY)  # a matrix
        rho[j] = R[0,1]

    # retain only the "good" predictors in the model
    # "good" is taken to mean: corr coef >= threshold


    
    indices = np.argsort(rho)[-int(len(rho)/10):]
    #indices = np.where(abs(rho)>=threshold)  # aight. The output of argwhere is not suitable for indexing arrays. For this purpose use nonzero(a) instead.
    
    #temp = np.transpose(trainX) #this sorta fixes it
    # trainX = np.transpose(temp[indices])
    trainX = trainX[::,indices]  #[0]]]
    trainX = (trainX - np.mean(trainX,axis=0))/np.std(trainX,axis=0)  # normalize
    # convert to z-score so that Euclidean distance doesn't
    # give more weight to predictors with a higher standard deviation
    
 
    # temp = np.transpose(testX) #this sorta fixes it
    # testX = np.transpose(temp[indices])
    testX = testX[::,indices] #[0]]
    testX = (testX - np.mean(testX,axis=0))/np.std(testX,axis=0)  # normalize
    
    
    #trainX has too many dimensions in line 9x. KNN is having trouble
    # np.shape(trainX)
    # (45, 1, 1)
    # test
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(trainX,trainY)
    y_hat = neigh.predict(testX)
    percent[i] = sum(testY != y_hat)/num_test # print((1*(testY != y_hat))/num_test)
    

pred_error_right_way = np.mean(percent)
print(pred_error_right_way)


#----------------------------------------------------------
# wrong way of doing cross-validation
#----------------------------------------------------------

# testing for univariate correlation

# threshold = 0.1  # threshold used to determine which predictors will be left in the model
rho = np.zeros(p)

for j in np.arange(0,p):
    x = X[:,j]  # a column consisting of a single predictor
    R = np.corrcoef(x,Y)  # returns a matrix
    rho[j] = R[0,1]

#indices = np.where(abs(rho)>=threshold)
indices = np.argsort(rho)[-int(len(rho)/10):]

# (erroneously) estimate generalization error by faux cross-validation

X = X[::,indices]  # it's ok to mute X now since this is the last time it's being used

# 
# X and Y were already shuffled earlier

for i in np.arange(0,num_folds):
    # randomly permutate before splitting into 10 ?
    # no because it's already pretty randomly arranged ?
    test_row_indices = np.arange(i*num_test,(i+1)*num_test)  # the rows that for the hold-out set
    testX = X[test_row_indices]
    testY = Y[test_row_indices] 
    
    trainX = np.delete(X,test_row_indices,axis=0)  # don't leave the hold-out set in the training set
    trainY = np.delete(Y,test_row_indices)
    
    # use only the "good" predictors in the model
    # trainX = trainX[::,indices[0]]
    trainX = (trainX - np.mean(trainX,axis=0))/np.std(trainX,axis=0)  # normalize 
    # testX = testX[::,indices[0]]
    testX = (testX - np.mean(testX,axis=0))/np.std(testX,axis=0)  # normalize
    
    # test
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(trainX,trainY)
    y_hat = neigh.predict(testX)
    percent[i] = sum(testY != y_hat)/num_test # print((1*(testY != y_hat))/num_test)


pred_error_wrong_way = np.mean(percent)
print(pred_error_wrong_way)



# < markdowncell>