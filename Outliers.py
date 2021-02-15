import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_ages(mu=50, sigma=13, num_samp=100,seed=42):
    np.random.seed(seed)
    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samp)
    sample_ages = np.round(sample_ages,decimals=0)
    return sample_ages

# creating normal distribution of ages for outlier model test
sample = create_ages()
#sns.boxplot(sample)
#plt.show()

ser = pd.Series(sample)
#explore the IQR values
ser.describe()
# 75% - 25% quartile range -- mannual vs numpy method
#IQR = 55.25 - 42.0
Quart = np.percentile(sample, [75,25])
IQR = Quart[0] - Quart[1]
# creating lower limit for age in the generated distribution
lower_limit = 42.0 - 1.5*(IQR)
#print(lower_limit)

df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/09-Feature-Engineering/../DATA/Ames_Housing_Data.csv')
corr = df.corr()['SalePrice'].sort_values()
#print(corr)

sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
plt.show()

