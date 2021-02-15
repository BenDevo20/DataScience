import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# import advertising data from file directory as dataframe
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Advertising.csv')

# standard train test split procedure
X = df.drop('sales', axis=1)
y = df['sales']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=101)

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create model
model = Ridge(alpha=100)
# model is estimates, X, y, negative mse (larger is better)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

# create model
model2 = Ridge(alpha=1)
# model is estimates, X, y, negative mse (larger is better)
scores2 = cross_val_score(model2, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

# compare the means of the errors that are produced to find the best model
scored1 = abs(scores.mean())
scored2 = abs(scores2.mean())

# after determing which alpha value has the better score fit the model to the data
model2.fit(X_train, y_train)
y_fin_pred = model.predict(X_test)
# final mean squared error on - tested on data that has never been seen
MSE = mean_squared_error(y_test, y_fin_pred)


