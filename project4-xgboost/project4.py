import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

# A simple helper function to generate train, validate, and test
def train_valid_test_split(dataX, dataY, train_ratio, valid_ratio, test_ratio, rand_seed=0):
    if(train_ratio + valid_ratio + test_ratio) > 1:
        return None, None, None, None, None, None
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(dataX, dataY, test_size=(valid_ratio + test_ratio), random_state=rand_seed)
        if not valid_ratio == 0:
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=(test_ratio / (test_ratio + valid_ratio)), random_state=rand_seed)
        else:
            test_x = valid_x
            test_y = valid_y
            valid_x = None
            valid_y = None
        return train_x, valid_x, test_x, train_y, valid_y, test_y

def predict_scores(mod, test_x, test_y):
    predictY = mod.predict(test_x)
    cmatrix = confusion_matrix(test_y, predictY, labels=[1, 0])
    score = accuracy_score(test_y, predictY)
    cm_key = np.array([['TP', 'TN'] , ['FP', 'FN']])
    print(cm_key)
    print(cmatrix)
    print(score)

def xgb_class_grid(param_dict, cv_func, train_x, train_y):
    clf = xgb.XGBClassifier()#tree_method='gpu_hist', predictor='gpu_predictor')
    mod = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=10)
    mod.fit(train_x, train_y)
    print(mod.best_params_)

    clf.set_params(**mod.best_params_)
    clf.fit(train_x, train_y)
    return clf

data = pd.read_csv('cardio_train.csv', delimiter=';')
data = data.sample(frac=1, random_state=0).reset_index(drop=True) # shuffle data in place

# testing code - to make sure the that the writes were happening
pd.set_option('max_columns', 13)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=np.inf)
data = data.drop(columns=['id'])
print(data)

# 250 seems to be what normal folks consider an overly high bp
max_bp = 250
clean_data = data[(data['ap_hi'] < max_bp) & (data['ap_lo'] < max_bp)]

# make these values 0/1 instead of 1/2
clean_data.loc[:, ['gender', 'cholesterol', 'gluc']] -= 1

# rescale the columns that are not 0-1. Returns np arrays
std_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(clean_data)
clean_data_np = scaler.transform(clean_data)

# grab the feature columns
# X = clean_data.iloc[:, :-1].to_numpy()
# y = clean_data['cardio'].to_numpy()
X = clean_data_np[::, :-1]
y = clean_data_np[::, -1]

print(X[:5])
print(y[:5])
# for arr in X:
#     if arr[0] == 0:
#         print(arr)

#use the np arrays to get the splits
trainX, validX, testX, trainY, validY, testY = train_valid_test_split(X, y, 0.8, 0.1, 0.1)
#xgb_class_grid()
#
# plt.bar(clean_data.columns.values[1:], model.feature_importances_)
# plt.xticks(rotation=45, ha="right")
# plt.show()

#reconvert back into dataframe and get dMatrices
trainX = pd.DataFrame(trainX, columns=clean_data.columns[:-1])
validX = pd.DataFrame(validX, columns=clean_data.columns[:-1])
testX = pd.DataFrame(testX, columns=clean_data.columns[:-1])

params = {
        "eta": list(np.arange(0.01,0.4,0.1)),
        # "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        # "min_child_weight": [1, 3, 5, 7],
        "lambda": list(np.arange(4, 4.8, 0.1)),
        "gamma": list(np.arange(0.5,0.7,0.1))
        # "colsample_bytree": [0.25, 0.5, 0.75, 1]
}

classifier = xgb_class_grid(params, GridSearchCV, trainX, trainY)
xgb.plot_importance(classifier)
plt.show()

predict_scores(classifier, testX, testY)

pickle.dump(classifier, open('class.p','wb'))

classifier.save_model('classifier.json')



model = xgb.XGBClassifier()
model.fit(trainX, trainY)
predict_scores(model, testX, testY)



#


# dtrainX = xgb.DMatrix(trainX, label=trainY)
# dvalidX = xgb.DMatrix(validX, label=validY)
# dtestX = xgb.DMatrix(testX, label=testY)
#
# param = {'max_depth': 4,
#          'eta': 0.3,
#          'objective': 'binary:logistic',
#          }  # 'num_class': 2 multi:softprob, binary:logistic
#
# boost_rounds = 100
#
# model2 = xgb.train(param, trainX,num_boost_round=boost_rounds,evals=[(testX, "Test")])
# #model2.feature_names = clean_data.columns.values[1:]
# xgb.plot_importance(model2)
# plt.show()
#
# predictY = model2.predict(testX)
#
# # labels [1,0] is important to order the confusion matrix as
# # TP TN
# # FP FN
# cmatrix = confusion_matrix(testY, predictY, labels=[1,0])
# score = accuracy_score(testY, predictY)
#
# print(cmatrix)
# print((cmatrix[0][0] + cmatrix[1][1])/np.sum(cmatrix) * 100)
# print(score)
