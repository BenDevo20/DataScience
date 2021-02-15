import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load

# importing data for regression function
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/08-Linear-Regression-Models/Advertising.csv')

# want to look at each advertising channel and the corresponding sales
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))
# TV spend vs Sales
axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].set_ylabel('sales')
axes[0].set_title('TV Spend')
# Radio spend vs Sales
axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].set_ylabel('sales')
axes[1].set_title('Radio Spend')
# Newspaper spend vs Sales
axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].set_ylabel('sales')
axes[2].set_title('Newspaper Spend')
plt.tight_layout()
#plt.show()

# creating features for model -- droppping one column in Dataframe that dont want
X = df.drop('sales', axis=1)
y = df['sales']

# running train test split regression
# test size - percent of data that should go to the test set / rand state - arbitrary number - same for all algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# creating regression function using scikitlearn
model = LinearRegression()
model.fit(X_train, y_train)
# output test prediction values from test data that was trained on the train data
test_pred = model.predict(X_test)
# compare prediction values to the y-test data (y-test are the real values of the output)
MAE = mean_absolute_error(y_test, test_pred)
MSE = mean_squared_error(y_test, test_pred)
RMSE = np.sqrt(mean_squared_error(y_test, test_pred))
print('Error Metrics: ', 'MAE:', MAE, 'MSE:', MSE, 'RMSE:',RMSE)

# evaluating the residuals - error between true y and the predicted y
test_residuals = y_test - test_pred
# plot residuals to identify any missed patterns
sns.scatterplot(x=y_test, y=test_residuals)
#plt.show()

# creating a deployable model
deploy_model = LinearRegression()
deploy_model.fit(X,y)
#print(deploy_model.coef_)
y_hat = deploy_model.predict(X)

fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))
# TV spend vs Sales
axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].plot(df['TV'],y_hat,'o', color='red')
axes[0].set_ylabel('sales')
axes[0].set_title('TV Spend')
# Radio spend vs Sales
axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].plot(df['radio'],y_hat,'o', color='red')
axes[1].set_ylabel('sales')
axes[1].set_title('Radio Spend')
# Newspaper spend vs Sales
axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].plot(df['newspaper'],y_hat,'o', color='red')
axes[2].set_ylabel('sales')
axes[2].set_title('Newspaper Spend')
plt.tight_layout()
plt.show()

# downloadable trained model file
dump(deploy_model, 'RegressModelSales.joblib')
loaded_model = load('RegressModelSales.joblib')

# testing loaded model on a new data set
# 149 on TV, 22 Radio, 12 newspaper -- what are the expected sales
campaign =[[149, 22, 12]]
mod = loaded_model.predict(campaign)
print(mod)
