import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from pytictoc import TicToc
import pickle
import random

random.seed(0)
np.random.seed(0)

# A simple helper function to generate train, validate, and test
def train_valid_test_split(data_x, data_y, train_ratio, valid_ratio, test_ratio, rand_seed=0):
    if(train_ratio + valid_ratio + test_ratio) > 1:
        return None, None, None, None, None, None
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=(valid_ratio + test_ratio), random_state=rand_seed)
        if not valid_ratio == 0:
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=(test_ratio / (test_ratio + valid_ratio)), random_state=rand_seed)
        else:
            test_x = valid_x
            test_y = valid_y
            valid_x = None
            valid_y = None
        return train_x, valid_x, test_x, train_y, valid_y, test_y

def predict_scores(clf_name, mod, test_x, test_y):
    predict_y = mod.predict(test_x)
    cmatrix = confusion_matrix(test_y, predict_y, labels=[1, 0])
    score = accuracy_score(test_y, predict_y)
    cm_key = np.array([['TP', 'TN'] , ['FP', 'FN']])
    print(clf_name, "confusion matrix:")
    print(np.c_[cm_key,cmatrix])
    print(clf_name, 'accuracy percentage: ', score*100)

def forest_clf(method, param_dict, cv_func, train_x, train_y, num_folds=10, rand_seed=0, num_iter=100):
    clf = method()
    if cv_func == GridSearchCV:
        cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds)
    elif cv_func == RandomizedSearchCV:
        cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds, random_state=rand_seed, n_iter=num_iter)
    cv_model.fit(train_x, train_y)
    print('Best params', cv_model.best_params_)
    print('Best score:', cv_model.best_score_)

    clf.set_params(**cv_model.best_params_)
    clf.fit(train_x, train_y)
    return clf

def xgb_clf(param_dict, cv_func, train_x, train_y, num_folds=10, gpu=None, rand_seed=0, num_iter=100):
    if gpu:
        clf = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        if cv_func == GridSearchCV:
            cv_model = cv_func(clf, param_dict, scoring='accuracy', cv=num_folds)
        elif cv_func == RandomizedSearchCV:
            cv_model = cv_func(clf, param_dict, scoring='accuracy', cv=num_folds, random_state=rand_seed, n_iter=num_iter)
    else:
        clf = xgb.XGBClassifier()
        if cv_func == GridSearchCV:
            cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds)
        elif cv_func == RandomizedSearchCV:
            cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds, random_state=rand_seed, n_iter=num_iter)
    cv_model.fit(train_x, train_y)
    print('Best params', cv_model.best_params_)
    print('Best score:', cv_model.best_score_)

    clf.set_params(**cv_model.best_params_)
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

# grab the feature columns -
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
trainX, _, testX, trainY, _, testY = train_valid_test_split(X, y, 0.9, 0, 0.1)

# forest_params = {
#     'n_estimators' : [10,100,1000],
#     'criterion' : ['gini','entropy'],
#     'max_depth' : list(range(1,11,1)),
#     'min_samples_split' : [2,3,4],
#     'min_samples_leaf' :  [2,3,4],
#     #'n_jobs' : [-1],
# }

# {'n_estimators': 100, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_depth': 10, 'criterion': 'gini'}

forest_params = {
    'n_estimators' : [1000],
    'criterion' : ['entropy'],
    'max_depth' : [8],
    #'min_samples_split' : [2],
    #'min_samples_leaf' :  [2],
    #'n_jobs' : [-1],
}

t = TicToc()
t.tic()
forest_model = forest_clf(RandomForestClassifier, forest_params, GridSearchCV, trainX, trainY)
t.toc()

predict_scores('Random Forest', forest_model, testX, testY)

pickle.dump(forest_model, open('random_forest_classifier.p', 'wb'))
#
# # mmodel = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=1000, n_jobs=16, random_state=0)
# # mmodel.fit(trainX, trainY)
# # predictions = mmodel.predict(testX)
# # sscore = accuracy_score(testY, predictions)
# # print(sscore)

fig, ax = plt.subplots()

forest_feature_ind = np.argsort(forest_model.feature_importances_)
forest_feature_ind = forest_feature_ind[::-1]
forest_feature_name = np.array(clean_data.columns.values[:-1])
forest_feature_name = forest_feature_name[forest_feature_ind]
forest_feature_val = forest_model.feature_importances_[forest_feature_ind]

ax.barh(forest_feature_name, forest_feature_val)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance of Random Forest Classifier')
# plt.xticks(rotation=45, ha="right")
plt.grid()
plt.show()


#xgb_class_grid()
#
# plt.bar(clean_data.columns.values[1:], model.feature_importances_)
# plt.xticks(rotation=45, ha="right")
# plt.show()

#reconvert back into dataframe
trainX = pd.DataFrame(trainX, columns=clean_data.columns[:-1])
#validX = pd.DataFrame(validX, columns=clean_data.columns[:-1])
testX = pd.DataFrame(testX, columns=clean_data.columns[:-1])


xgb_params = {
        "booster": ['dart'],
        "objective": ["binary:logistic", "reg:tweedie", "reg:gamma", "rank:pairwise"], #try various objectives
        "nthread": [-1], #use as many threads as available
        "eta": list(np.arange(0,10,0.01)),
        # "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        # "min_child_weight": [1, 3, 5, 7],
        "lambda": list(np.arange(0, 10, 0.05)), # [4.7]
        "gamma": list(np.arange(0, 10, 0.05)),
        #"colsample_bytree": list(np.arange(0,1.01,0.1)), #[0.25, 0.5, 0.75, 1]
}

#{'objective': 'binary:logistic', 'nthread': -1, 'lambda': 2.0500000000000003, 'gamma': 3.4000000000000004, 'eta': 0.02, 'booster': 'dart'}
# best performance:{'eta': 0.31000000000000005, 'gamma': 0.6, 'lambda': 4.299999999999999}
best_xgb_params = {
    'objective' : 'binary:logistic',
    'eta' : 0.31,
    'gamma' : 0.6,
    'lambda' : 4.29,
}

t1 = TicToc()

t1.tic()
xgb_model = xgb_clf(xgb_params, RandomizedSearchCV, trainX, trainY)
t1.toc()

pickle.dump(xgb_model, open('xgb_class.p', 'wb'))

xgb_model.save_model('classifier.json')
predict_scores("xgb RandomizedSearchCV model", xgb_model, testX, testY)
xgb.plot_importance(xgb_model, importance_type='gain')
plt.show()


t2=TicToc()
t2.tic()
best_xgb_model = xgb.XGBClassifier(**best_xgb_params)
best_xgb_model.fit(trainX, trainY)
t2.toc()
predict_scores("best xgb params found using GridSearchCV", best_xgb_model, testX, testY)
xgb.plot_importance(best_xgb_model,importance_type='gain')
plt.show()

t3=TicToc()
t3.tic()
vanilla_xgb_model = xgb.XGBClassifier()
vanilla_xgb_model.fit(trainX, trainY)
t3.toc()
predict_scores("Vanilla xgb", vanilla_xgb_model, testX, testY)
xgb.plot_importance(vanilla_xgb_model,importance_type='gain')
plt.show()


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
