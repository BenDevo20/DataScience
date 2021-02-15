"""
simple linear regression - regression function with only one feature
create best fit line to map out linear relationship
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing excel file that has data for regression function
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/08-Linear-Regression-Models/Advertising.csv')

# finding relationship between advertising and sales
# creating total spend column in dataframe for single feature in regression function
df['totSpend'] = df['TV'] + df['radio'] + df['newspaper']

"""
# plotting data to see visualize potential relationship
sns.scatterplot(data=df, x='totSpend', y='sales')
# plotting simple regression using OLS
sns.regplot(data=df, x='totSpend', y='sales')
plt.show()
"""
# creating beta values for function
X = df['totSpend']
y = df['sales']

# compute OLS using numpy - degree = 1 because it is simple linear regression
beta = np.polyfit(X, y, deg=1)
# calculating predicted sales for potential spend values
pot_spend = np.linspace(0,500,100)
predict_sales = beta[0] * pot_spend + beta[1]
# plt calculated predicted line over the real sales data to see trend fit
sns.scatterplot(x='totSpend', y='sales', data=df)
plt.plot(pot_spend, predict_sales, color='r')
#plt.show()

# using regression for a given budget of 200 marketing dollars
spend = 200
predict_sale = beta[0]*spend + beta[1]
#print(predict_sale)

# fitting for higher degree polynomials using np.polyfit
# y = b3x**3 + bx2*x**2 + b1x + b0
np.polyfit(X,y,3)

