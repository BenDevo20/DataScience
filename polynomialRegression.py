"""
address non-linear relationship between 2 features where linear regression will be inaccurate to model
interaction terms - two features are correlated - the effectiveness of one is predicated on the other
    mulitply two features together to create interaction feature
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

# importing model data
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/08-Linear-Regression-Models/Advertising.csv')

# creating features for polynomial regression model
X = df.drop('sales', axis=1)
y = df['sales']

# what degree of the polynomial regression function - interaction terms are included in the regression
poly_convert = PolynomialFeatures(degree=2, include_bias=False)
# first 3 are normal features, next 3 are the interaction terms, next 3 are the squared terms
poly_features = poly_convert.fit_transform(X)

# train test split on the 9 features created by polynomial function
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
test_predict = model.predict(X_test)

# test statistics
MAE = mean_absolute_error(y_test, test_predict)
MSE = mean_squared_error(y_test, test_predict)
RMSE = np.sqrt(mean_squared_error(y_test, test_predict))

# loop to figure out what the best polynomial degree is to pick - this is testing all potential over and under fit
train_rmse = []
test_rmse = []

for d in range(1,10):
    poly_convert = PolynomialFeatures(degree=d,include_bias=False)
    poly_features = poly_convert.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=42)
    model = LinearRegression()
    model.fit(X_train,y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_RMSE = np.sqrt(mean_squared_error(y_train, train_pred))
    test_RMSE = np.sqrt(mean_squared_error(y_test, test_pred))

    train_rmse.append(train_RMSE)
    test_rmse.append(test_RMSE)

plt.plot(range(1,6), train_rmse[:5], label='Train RMSE')
plt.plot(range(1,6), test_rmse[:5], label='Test RMSE')
plt.show()

deployPolyConvert = PolynomialFeatures(degree=3, include_bias=False)
deployPoly = LinearRegression()
full_deploy = deployPolyConvert.fit_transform(X)
full_deploy.fit(full_deploy, y)
dump(deployPoly, 'deploy_poly.joblib')
dump(deployPolyConvert, 'deploy_polyConvert.joblib')

