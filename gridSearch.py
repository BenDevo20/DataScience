import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# read in csv file to df
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Advertising.csv')

# create regression features
X = df.drop('sales', axis=1)
y = df['sales']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# grid search looks for the best values to tune the model to
base_elastic = ElasticNet()
# dictionary of paramters that are going to passed through the model to find the best parameters
# look at documentation for ranges and understanding of the parameters
param_grid = {'alpha': [0.1, 1, 5, 10, 50, 100], 'l1_ratio': [.1, .5, .7, .95, .99, 1]}
# performing gridsearch
grid_mod = GridSearchCV(estimator=base_elastic, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=.5)
# fit the model to training sets
grid_mod.fit(X_train, y_train)
# outputting best estimators for l1 ratio and alpha value
print(grid_mod.best_estimator_)

# create dataframe of results
results = pd.DataFrame(grid_mod.cv_results_)
print(results)

