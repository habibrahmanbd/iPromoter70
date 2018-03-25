from __future__ import division
import random

# Author import
#====================================================================
print 'Author: Md.Siddiqur Rahman'
print 'Email : ronicse59@gmail.com'
print('___________________________________________________')

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For testing puposes, start always from 123, no random state.
# For random value just comment the line below
#====================================================================
np.random.seed(seed=123)

# scikit-learn library import
#====================================================================
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# Necessary Library for Calculation
#====================================================================
from sklearn.metrics import accuracy_score, log_loss, \
    classification_report, confusion_matrix, roc_auc_score, \
    average_precision_score, f1_score

# Input benchmark data and output features as "Features.csv"
#====================================================================
input_file_name = "BenchmarkData.txt"
features_file_name = "Features.csv"


# Here we can generate our features set using different length
#====================================================================
if(1):
    # There are 11 sets of features
    # all 1 meaning that we wants to extract all the 11 sets of features
    # please ignore the first one.
    # if we want to stop any feature, make it 0 from feature_list
    feature_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    import feature_extractor
    feature_extractor.generator(input_file_name, features_file_name, feature_list)
# Load the Featureset:
#====================================================================
D = pd.read_csv(features_file_name, header=None)
D = D.drop_duplicates()  # Return : each row are unique value

# Divide features (X) and classes (y) :
#====================================================================
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

print '-> Total Features: ',len(X[0])


# Encoding y :
#====================================================================
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)


# A list of classifiers
#====================================================================
Classifiers = [
    # comment classifiers those you don't want to run.
    #----------------------------------------------------------------
    SVC(kernel='rbf', C=4, probability=True, decision_function_shape='ovo', tol=0.1, cache_size=200),
    SVC(kernel='rbf', degree=4, decision_function_shape='ovo', tol=0.1, cache_size=100, gamma=13, probability=True, C=4),
    XGBClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(n_jobs=1000),
    KNeighborsClassifier(n_jobs=500),
    DecisionTreeClassifier(),
    GaussianNB(),
    BaggingClassifier(),
    RandomForestClassifier(n_estimators=500),
    AdaBoostClassifier(n_estimators=500),
    GradientBoostingClassifier(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(learning_rate='adaptive'),
]

# For 10 fold, spliting with 10-FCV :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Pick all classifier within the Classifier list and test one by one
#====================================================================
# CM = Confusion Matrix
print '-> Start classification   ...'
CM = np.zeros((2,2), dtype=int)
fold = 1
for classifier in Classifiers:
    accuracy = []
    auroc = []
    aupr = []
    F1 = []
    print('___________________________________________________')
    print('Classifier: '+classifier.__class__.__name__)
    model = classifier
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]


        # Scaling the feature
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scale = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)

        print 'F%d,'%fold,
        fold += 1

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy.append(accuracy_score(y_pred=y_pred, y_true=y_test))
        auroc.append(roc_auc_score(y_true=y_test, y_score=y_proba))
        aupr.append(average_precision_score(y_true=y_test, y_score=y_proba))
        F1.append(f1_score(y_pred=y_pred, y_true=y_test))

        CM += confusion_matrix(y_pred=y_pred, y_true=y_test)

    print ''

    TN, FP, FN, TP = CM.ravel()

    print('---------------------------------------------------')
    print '| Acc  | ROC  |APUR  | Sp   | Sn   | MCC  | F1   |'
    print('---------------------------------------------------')
    print '|%.3f ' % np.mean(accuracy)+'|%.3f ' % np.mean(auroc) +'|%.3f ' % np.mean(aupr) \
    + '|%.3f ' % (TN / (TN + FP)) +'|%.3f ' % (TP / (TP + FN)) \
    + '|%.3f ' % ((TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))) \
    + '|%.3f |' % np.mean(F1)

    print('---------------------------------------------------')
    #print 'Confusion Matrix:\n', CM
    #print TN, FP, FN, TP
    #CM = np.zeros((2, 2), dtype=int)