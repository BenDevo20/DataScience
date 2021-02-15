import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# import advertising data from file directory as dataframe
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Advertising.csv')

# standard train test split procedure
X = df.drop('sales', axis=1)
y = df['sales']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create models to check hyperparameters
model1 = Ridge(alpha=100)
model2 = Ridge(alpha=1)

# can pass in scoring metrics explained in scikit learn documentation
scores1 = cross_validate(model1, X_train, y_train, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'], cv=10)
scores2 = cross_validate(model2, X_train, y_train, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'], cv=10)

# transform scores dictionary into dataframe
scores1 = pd.DataFrame(scores1)
scores1.mean()
scores2 = pd.DataFrame(scores2)
scores2.mean()

model2.fit(X_train, y_train)
y_fin_pred = model2.predict(X_test)
MSE2 = mean_squared_error(y_test, y_fin_pred)

