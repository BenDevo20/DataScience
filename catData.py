import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Ames_Housing_Feature_Description.txt', 'r') as f:
    print(f.read())

# create data frame from csv file
df = pd.csv_read('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/Ames_NO_MISSING_DATA.csv')

# applying numeric value and making it a str
df['MS Subclass'] = df['MS Subclass'].apply('str')

direction = pd.series(['up','up','down'])
# create dummy variables from pd.series for one-hot encoding
# can pass in list of columns to create one-hot encoding
pd.get_dummies(direction,drop_first=True)

# checking the data type for each of the columns -- object = string
x_obj = df.select_dtypes(include='object')
# create numeric df - exclude all objects which are read as strings
x_num = df.select_dtypes(exclude='object')
# passing this through get dummies - turns strings into one-hot for every category
df_objects_dummies = pd.get_dummies(x_obj, drop_first=True)

fin_df = pd.concat([df_objects_dummies, x_num], axis=1)




