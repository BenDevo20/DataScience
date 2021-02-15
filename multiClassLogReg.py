import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

# read in csv file
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/iris.csv')

# distribution of category counts
df['species'].value_counts()

X = df.drop('species', axis=1)
y = df['species']

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=101)

# scale features
scaler = StandardScaler()
scaled_Xtrain = scaler.fit_transform(X_train)
scaled_Xtest = scaler.transform(X_test)

# saga supports elastic net, multi class - binary is set for individual labels
log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)
# add in things to search for
penalty = ['11', '12', 'elasticnet']
l1_ratio = np.linspace(0, 1, 20)
# how strong should the penalty be
C = np.log(0, 10, 20)
# dictionary labels need to match what the algorithm in the documentation is expecting
param_grid = {'penalty': penalty, 'l1_ratio': l1_ratio, 'C': C}
# creating grid search with log model and param dictionary
grid_model = GridSearchCV(log_model, param_grid=param_grid)
grid_model.fit(scaled_Xtrain, y_train)

# performance metrics
y_pred = grid_model.predict(scaled_Xtest)
acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
print(class_rep)

