import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb


data = pd.read_csv('cardio_train.csv', delimiter=';')

# testing code - to make sure the that the writes were happening
#pd.set_option('max_columns', 13)
#print(data[(data['ap_hi'] < 250) & (data['ap_lo'] < 250)].head(n=1877))
#print(data[data['ap_hi'] > 250])


# 250 seems to be what normal folks consider to be a high bp
max_bp = 250
clean_data = data[(data['ap_hi'] < max_bp) & (data['ap_lo'] < max_bp)]
clean_data.loc[:, ['gender', 'cholesterol', 'gluc']] -= 1
print(clean_data.head(5))
#print(clean_data.tail())

#grab the feature columns
X = clean_data.iloc[:, :-1].to_numpy()
y = clean_data['cardio'].to_numpy()

print(X[:5])
print(y[:5])
# for arr in X:
#     if arr[0] == 0:
#         print(arr)





