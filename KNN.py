"""
k nearest neighbors - assigns a label to new data based on the distance between the old data and new data
what points are the new point closest to
    how many points should you consider when determining classification
    find the closest historical label point(s) depending on what k you set
    when you have multi connecting points you want to take the majority points
    when there is a tie - always choose odd Ks, random break tie
    choose nearest class point to break tie, which one when k=1
in scikitlearn the ordered data matters and which label it went to first is the tiebreaker

reducing error
1. elbow method - manual method - train model with multiple k values and mark down error rate for each k value
    you will eventually begin to increase k to the point that it is no longer significant
        computationally and complexity may be an overkill
2. grid search

"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_confusion_matrix

# read in csv file 
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/gene_expression.csv')

# visualizing data -- alpha adds in transparency to see stacked points
sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue='Cancer Present', alpha=0.5)
# define limits to see where the points begin to cross over
plt.xlim(2,6)
plt.ylim(4,8)
#plt.show()

# define features
X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

# create split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=101)
# always scale data for k nearest
scaler = StandardScaler()
scaled_Xtrain = scaler.fit_transform(X_train)
# dont fit on the test set - avoid data leaks
scaled_Xtest = scaler.transform(X_test)

# simplest KNN model
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(scaled_Xtrain, y_train)

# evaluate model
y_pred = knn_model.predict(scaled_Xtest)
conf_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)

