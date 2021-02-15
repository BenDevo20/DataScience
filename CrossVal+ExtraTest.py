import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# import advertising data from file directory as dataframe
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Advertising.csv')

# standard train test split procedure
X = df.drop('sales', axis=1)
y = df['sales']

# creating extra test split to adjust for data leakage
X_train, X_other, y_train, y_other = train_test_split(X,y,test_size=.30, random_state=101)
# split again for validation - splitting only the test size from previous split
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=.50, random_state=101)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
# three splits for cross validation 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

model1 = Ridge(alpha=100)
model2 = Ridge(alpha=1)

# model 1 performance
model1.fit(X_train, y_train)
test_pred1 = model1.predict(X_eval)
MSE1 = mean_squared_error(y_eval, test_pred1)

# model 2 performance
model2.fit(X_train, y_train)
test_pred2 = model1.predict(X_eval)
MSE2 = mean_squared_error(y_eval, test_pred1)

# satisified wth MSE and now testing on final data split that has not been used
y_fin_test = model2.predict(X_test)
MSE_fin = mean_squared_error(y_test, y_fin_test)

