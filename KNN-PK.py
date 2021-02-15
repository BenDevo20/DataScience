import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

# want to keep track of test error and find k that minimizes error
test_error_rate = []

for k in range(1, 30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_Xtrain, y_train)

    y_pred = knn_model.predict(scaled_Xtest)
    # find acc score for each k in for loop
    acc_score = accuracy_score(y_test, y_pred)
    # 1 - accuracy is the percentage of error
    test_error = 1 - acc_score
    # append the error to the error rate list outside for loop
    test_error_rate.append(test_error)

# index position will be the number of ks in model -- find where the error rate levels off
print(test_error_rate)
# visualize the error rate
plt.plot(range(1,30), test_error_rate)
plt.ylabel('error rate')
plt.xlabel('K neighbors')

"""
it is not best practice to pick where the min error rate - have to consider the percentage jump with ever additional 
k in model 
find the point where the percent decrease is error is close to the error rate with 1 less k 
"""

# setting up pipeline - grid search using cross validation
scaler = StandardScaler()
knn = KNeighborsClassifier()
# set up operations - string code that matches param keys from documentation
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)
# perform grid search with cross validation
k_values = list(range(1,20))
# set up param grid - wording has to be specific because the grid search wont know what schema to fit
# syntax is from stack overflow
param_grid = {'knn__n_neighbors': k_values}
# setup grid search
full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
# fitting data - do not need to scale because the pipe scales it for you
full_cv_classifier.fit(X_train, y_train)
# which parameter did the pipe choose for its kvalue
print(full_cv_classifier.best_estimator_.get_params)

# final training on hold out set and looking at the classification report
fin_pred = full_cv_classifier.predict(X_test)
print(classification_report(y_test, fin_pred))

