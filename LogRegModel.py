import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# read in csv file
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/hearing_test.csv')

# separating out features for regression model 
X = df.drop('test_result', axis=1)
y = df['test_result']

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)
scaler = StandardScaler()
# fit and transformation in one step only fit on the training data
scaled_xTrain = scaler.fit_transform(X_train)
scaled_xTest = scaler.transform(X_test)

log_model = LogisticRegression()
# fit to scaled data
log_model.fit(scaled_xTrain, y_train)
# coefficients
coef = log_model.coef_
# predictions are just ones and zeros
y_pred = log_model.predict(scaled_xTest)
# prediction of the probability that it will be in the right classification
y_pred_prob = log_model.predict_proba(scaled_xTest)

"""
build out a matrix for the potential errors 
confusion matrix 
    two columns will be the true values on back test data 
    two rows will be the predicted values 
    True/Positive - the predicted row and true column are the same 
    True/Negative - the predicted row and true column for other classification are the same 
    False/Negative - when the row and column dont match 
    False/Positive - when the row and column dont match

accuracy - how often is the model correct (sum of true positives and true negatives) / (total sample space) 
accuracy paradox - when there is an imbalanced label - most points fall under one classification and the model 
    continues to label all data points that one classification for predicted values 
    gives a high accuracy rating, but is misclassifying data points 
"""

"""
recall - sensitivity - when one of the new data points goes through model how often does the classification accurately 
    place those points in the right feature 
    answers the question - how many relevant cases are found 

precision - when the prediction is positive, how often is it correct 

F1 score - harmonic mean of precision and recall 
F = (2 * precision * recall) / (precision + recall)
"""

"""
ROC Curve
Receiver operator characteristic curve 
True Positive - y-axis ----- False positive - x-axis
there is a trade off between true positives and false positives 
default cut off point is 0.5 -- anything above will be positive and below negative
    take the ratio of the above and below  
as you lower the cutoff threshold you can create an imbalance for either false negatives or false positives 
    this is based on domain knowledge and it is more acceptable to have more of one than the other 
"""

# performance metrics for logistic regression
# given the features how well can the model predict future values classifications
acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
# plotting the confusion matrix - pass in the model and scaled data
# false negatives and false positives will be the top right and bottom left - should be smaller than other boxes
plot_confusion_matrix(log_model, scaled_xTest, y_test, normalize='all')
# classification report
class_rep = classification_report(y_test, y_pred)
# need to print report since it is a list of strings with new line calls
print(class_rep)
# precision score
prec_sco = precision_score(y_test, y_pred)
# acc score
acc_sco = accuracy_score(y_test, y_pred)

# plotting curves with model as estimator and scaled x data with y-test data
plot_roc_curve(log_model, scaled_xTest, y_test)
plot_precision_recall_curve(log_model, scaled_xTest, y_test)


