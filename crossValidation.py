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

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=101)
# scale data to make sure features are regularized
scaler = StandardScaler()
# only scale training data so there is no data leaks
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create model
model = Ridge(alpha=100)
model.fit(X_train, y_train)
# evaluate the model
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)

# create another model to find alpha - adjust hyper parameters
model2 = Ridge(alpha=1)
model2.fit(X_train, y_train)
y_pred2 = model.predict(X_test)
MSE2 = mean_squared_error(y_test, y_pred2)

"""
Advantage of train test split - separate into two sets to evaluate model - saves on computation time
    simple and readable 
disadvantage - hard to find the best hyper parameters - creating new model and checking alpha numbers
    MSE is not the fairest report - since it has seen X_test multiple times
    hyper parameters were adjusted on test set performance - hidden interaction
    want to hold another set of data that has not been seen by any models 
"""

