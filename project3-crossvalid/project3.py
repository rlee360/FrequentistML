# CODE REVIEW THURS 7/21

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random

random.seed(0)
np.random.seed(0)

# textbook's Gaussian random vectors

num_data = 50
p = 1

X = np.random.randn(num_data, p)   
#Y = np.random.randint(low=0, high=2, size=num_data)  # 50% 0's, 50% 1's
Y = random.shuffle(25*[0] + 25*[1])

# train_test_split we will need to use kfold instead
#train_test_split(dataX, dataY, test_size=(valid_ratio+test_ratio), random_state=rand_seed)


#---------------------------------------------
# Parameters for K-nearest neighbors

k = 1  # how many nearest neighbors to use in the k-nearest neighbors


#---------------------------------------------
# correct way of doing cross-validation
#---------------------------------------------

threshold = 0.1
percent = np.zeros(10)
nfolds = 10

for i in np.arange(0,nfolds):
    # randomly permutate before splitting into 10 ?
    one_fold = np.floor(num_data/nfolds)
    left_out_row_indices = np.arange(i*one_fold,(i+1)*one_fold)
    testX = X[left_out_row_indices,::]
    testY = Y[left_out_row_indices,::]
    
    # left_in_row_indices = np.arange(0,i*one_fold-1)
    temp = np.delete(data,row_indices,p)  # get rid of the left_out part
    X = temp[:,0:p-1]
    Y = temp[:,-1]

    # testing for univariate correlation

    rho = np.zeros(p)

    for j in np.arange(0,p):
        x = X[:,j]
    # CHANGE
        C = cov(x,Y)
        rho(j) = C(1,2)/sqrt(C(1,1)*C(2,2))

    indices = find(abs(rho)>=threshold)
    
    # use only the "good" predictors in the model
    X1 = X[:,indices]
    X1 = (X1 - mean(X1))/std(X1)  # normalize it (z-score)
    testX = testX[:,indices]
    testX = (testX-mean(testX))/std(testX)  # normalize
    
    # test
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X1,Y)
    y_hat = neigh.predict(testX)
    [~,percent(i)] = biterr(testY,y_hat)
    

pred_error_right_way = mean(percent)


#----------------------------------------------------------
# wrong way of doing cross-validation
#----------------------------------------------------------

# testing for univariate correlation

threshold = 0.1
rho = np.zeros(p)
X = data[:,0:p-1]
Y = data[:,-1]

for j in np.arange(0,p):
    x = X[:,j]
    C = cov(x,Y)
    rho(j) = C(1,2)/sqrt(C(1,1)*C(2,2))
end

indices = find(abs(rho)>=threshold)

# (erroneously) estimate generalization error by faux cross-validation

for i in np.arange(0,10):
    # randomly permutate before splitting into 10 ?
    # no because it's already pretty randomly arranged ?
    row_indices = 1+(i-1)*150:152+(i-1)*150
    left_out = data[row_indices,:]
    testX = left_out[:,0:p-1]
    testY = left_out[:,-1]  # same as [:,p]
    
    temp = data
    temp[row_indices,:] = []  # get rid of the left_out part
    X = temp[]:,0:p-1]
    Y = temp[:,-1]
    
    # use only the "good" predictors in the model
    X1 = X[:,indices]
    X1 = (X1 - mean(X1))/std(X1)  # normalize it (z-score)
    testX = testX[:,indices]
    testX = (testX-mean(testX))/std(testX)  # normalize
    
    # test
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X1,Y)
    y_hat = neigh.predict(testX)
    [~,percent(i)] = biterr(testY,y_hat)


pred_error_wrong_way = mean(percent)



# < markdowncell>