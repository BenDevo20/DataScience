import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import SCORERS
from sklearn.linear_model import Lasso, LassoCV

# create dataframe holding model data
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/08-Linear-Regression-Models/Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# create polynomial feature set
polyConvert = PolynomialFeatures(degree=3, include_bias=False)
polyFeatures = polyConvert.fit_transform(X)

# creating split data -- split on polyfeatures not the original X data
X_train, X_test, y_train, y_test = train_test_split(polyFeatures, y, test_size=0.3, random_state=101)

# scale the features based on mean or standard deviation on the train set - this avoids leakage
scaler = StandardScaler()
scaler.fit(X_train)
ScaledX_train = scaler.transform(X_train)
ScaledX_test = scaler.transform(X_test)

# creating ridge model
# use cross validation to find the best alpha value to optimize the model
ridge_model = Ridge(alpha=10)
ridge_model.fit(ScaledX_train, y_train)
test_predict = ridge_model.predict(X_test)

ridge_cv_model = RidgeCV(alphas=(0.1,1.0,10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(ScaledX_train, y_train)
test_predictCV = ridge_cv_model.predict(ScaledX_test)

# checking which alpha performed the best
print(ridge_cv_model.alpha_)
# this is basically a dictionary of error/performance metrics
SCORERS.keys()

# performance results
RCV_MAE = mean_absolute_error(y_test, test_predictCV)
RCV_RMSE = np.sqrt(mean_squared_error(y_test, test_predictCV))

# creating lasso model - epsilon = aplha min - alpha max ---- there is a default of 5 fold for cross validation
lasso_cvModel = LassoCV(eps=0.001, n_alphas=100, cv=5, max_iter=10000000)
lasso_cvModel.fit(ScaledX_train, y_train)
test_predictlasso = lasso_cvModel.predict(ScaledX_test)
# performance results
LassoMAE = mean_absolute_error(y_test, test_predictlasso)
LassoRMSE = np.sqrt(mean_squared_error(y_test, test_predictlasso))


