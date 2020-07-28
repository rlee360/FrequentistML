import pandas as pd
import numpy as np

data = pd.read_csv('cardio_train.csv', delimiter=';')

# testing code - to make sure the that the writes were happening
pd.set_option('max_columns', 13)
#print(data[(data['ap_hi'] < 250) & (data['ap_lo'] < 250)].head(n=1877))
#print(data[data['ap_hi'] > 250])


# 250 seems to be what normal folks consider to be a high bp
max_bp = 250
clean_data = data[(data['ap_hi'] < max_bp) & (data['ap_lo'] < max_bp)]
print(clean_data)

