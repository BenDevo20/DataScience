import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
with open('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Ames_Housing_Feature_Description.txt', 'r') as f:
    print(f.read())
"""
# importing housing data for model
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Ames_outliers_removed.csv')
df = df.drop('PID', axis=1)

# percent of missing information per feature
def percent_missing(df):
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan

# output the percent of missing data per feature in the data set
percent_nan = percent_missing(df)

# visualizing missing data
"""
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)
plt.show()
"""

