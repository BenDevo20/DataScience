"""
logistic regression is could for predicting categorical labels
logistic regression is a classification technique, not a traditional numeric regression analysis

works by transforming linear regression into a categorical classification regression
1/1+e^-x

exploratory data analysis (EDA)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in csv file
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/hearing_test.csv')

# looking at distribution of test results as a sum value of their classification (1-pass 0-did not pass)
sns.countplot(x='test_result',data=df)
# distribution of features per category
sns.boxplot(x='test_result', y='age', data=df)
plt.figure(dpi=150)
sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result', alpha=0.5)
sns.pairplot(df, hue='test_result')
# heat map of the correlations between the features
sns.heatmap(df.corr(), annot=True)
# plt.show()




