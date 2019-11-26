# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:55:30 2019

@author: willh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def get_data():
    x = input("Laptop or computer (l/c)?:")
    if x == 'l':
        os.chdir(r'C:\Codes\Cox_Manheim_analysis')
        data_dict = pd.read_excel('Case Study and Data Dictionary 201407.xlsx', sheet_name = 1, skiprows = 1)
        raw_data = pd.read_excel('transaction_data_processed.xlsx', sheet_name = 'data')
        with open(r'C:\Codes\Cox_Manheim_analysis\Case_Study.txt') as f:
            case_study = f.read()
    if x == 'c':
        os.chdir(r'C:\Users\willh\Cox_Manheim_Analysis\Car_Sales')
        data_dict = pd.read_excel(r'C:\Users\willh\Cox_Manheim_Analysis\Case Study and Data Dictionary 201407.xlsx', sheet_name = 1, skiprows = 1)
        raw_data = pd.read_excel(r'C:\Users\willh\Cox_Manheim_Analysis\Car_Sales\transaction_data_processed.xls')
        case_study = None
    return data_dict, raw_data, case_study

data_dict, raw_data, case_study = get_data()        

# Make copy of data
data = raw_data.copy()

import datetime as dt
data['DMSTDESL_DT'] = pd.to_datetime(data['DMSTDESL'], format = '%Y%m%d')
data['Sale_day_name'] = data['DMSTDESL_DT'].dt.weekday
# make a column for week of month (1st week, 2nd week, 3rd week, ...)
days = pd.DatetimeIndex(data['DMSTDESL_DT'].unique())
week_month = {}
week = 1
#datim = dt.datetime(2013,1,1)
#current_month = dt.datetime.timestamp(dt.datetime(2013,1,1)).month
current_month = dt.datetime(2013,1,1).month
for d in days:
#    print(int(d.month))
    if current_month == int(d.month):
        week_month[d] = week
        week += 1
    else:
        week = 1
        current_month = int(d.month)
        week_month[d] = week
        week +=1
        
data['month_week'] = data['DMSTDESL_DT'].map(week_month) 

# check out some columns
sale_days = data['Sale_day_name'].unique() # sales happen on wednesdays
sold = data[data['DMSOLD'] == 1]
sold['DMSTDESL_DT'].groupby(sold['DMSTDESL_DT']).count().plot.bar()
sold['DMSTDESL_DT'].value_counts().plot.bar()

pd.crosstab(data['DMMAKE'],data['DMSOLD']).plot.bar()
pd.crosstab(data['DMMODEL'],data['DMSOLD']).plot.bar()
pd.crosstab(data['DMJDCAT'], data['DMSOLD']).plot.bar()
pd.pivot_table(data, index=['DMMAKE', 'DMMODEL'], columns=['DMSOLD'], aggfunc=len)
data['DMSOLD'][data['DMSOLD']==1].groupby([data['DMMAKE'],data['DMMODEL']]).count()

c = data.columns
## Make categorical variables
#cat_cols = ['DMMONTH', 'SLANE_', 'DMOPCSUID', 'DMMAKE','STIMES_bin', 'Arbitration_bins','DMMODELYR_bins',]
#for col in cat_cols:
#    data[col] = data[col].astype('category')

# Drop columns that I won't be using
drop_cols = ['SFLOOR', 'DMECRDATE','DMSTDESL', 'DMSALWK', 'SSALE_', 'SRUN_', 'STIMES', 'DMTRANTYPE', 'DMSELLRNM', 'DMJDCAT', 
             'DMMODEL', 'DMBODY', 'SSER17', 'AGEDDAYS', 'DMPRECOND', 'DMPOSTCOND', 'DMDETFEE', 'DMRECONFEE','SFLR_lower','% Arbitration','imp_STIMES','VNMMR','SSLEPR'
             , 'DMOPCSUID','DMMODELYR','SMILES','DMMONTH','SLANE_','STIMES_bin3','STIMES_bin', 'DMSTDESL_DT','Sale_day_name']
data_penult = data.drop(drop_cols, axis =1)
data[data.columns != drop_cols]

# feature engineering
data_penult = data[['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins','DMSOLD']]
results1 = initial_all_results

data_penult = data[['SFLOOR_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins','DMSOLD']]
results2 = initial_all_results

data['STIMESxArb'] = data['STIMES_bin2'] + " " + data['Arbitration_bins']
ECRdata = data[data['SFLOOR_IND'] == 'Y']
data_penult = ECRdata[['STIMESxArb', 'DMSOLD', 'DMECRDATE_IND', 'DMMODELYR_bins']]
data_penult = ECRdata[['DMECRDATE_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins','DMSOLD']]
results3 = initial_all_results

data_penult = data[['STIMESxArb', 'DMSOLD', 'DMECRDATE_IND', 'DMMODELYR_bins']]
results4 = #could not complete - SVM

data['MODELYRxDMMODEL'] = data['DMMODELYR_bins'] + ' ' + data['DMMODEL']
data_penult = data[['MODELYRxDMMODEL', 'STIMESxArb','DMSOLD','DMECRDATE_IND']]

data['SFLOORxSTIMES'] = data['STIMES_bin2'] + ' ' + data['SFLOOR_IND']
data['SFLOORxDMECR'] = data['DMECRDATE_IND'] + ' ' + data['SFLOOR_IND']
data['SFLOORxARB'] = data['Arbitration_bins'] + ' ' + data['SFLOOR_IND']
data['SFLOORxDMMODEL'] = data['DMMODELYR_bins'] + ' ' + data['SFLOOR_IND']
data_penult = data[['SFLOORxSTIMES','SFLOORxDMECR', 'SFLOORxARB', 'SFLOORxDMMODEL','DMSOLD']]
results5 = initial_all_results

data['STIMESxSFLOOR'] = data['STIMES_bin2'] + ' ' + data['SFLOOR_IND']
data['STIMESxDMECR'] = data['STIMES_bin2'] + ' ' + data['DMECRDATE_IND']
data['STIMESxARB'] = data['STIMES_bin2'] + ' ' + data['Arbitration_bins']
data['STIMESxDMMODEL'] = data['STIMES_bin2'] + ' ' + data['DMMODELYR_bins']
data_penult = data[['STIMESxSFLOOR', 'STIMESxDMECR','STIMESxARB','STIMESxDMMODEL','DMSOLD']]

# Make dummy variables
cat_cols = data_penult.loc[:, data_penult.columns != 'DMSOLD'].columns
cat_cols = ['STIMESxArb', 'month_week']
for col in cat_cols:
    cat_list='var'+'_'+col
    cat_list = pd.get_dummies(data_penult[col], prefix=col, drop_first = True)
    data1=data_penult.join(cat_list)
    data_penult=data1
data_vars=data_penult.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_cols]
# final data columns will be:
data_final=data_penult[to_keep]
data_final.columns.values

# Make X and y datasets
X = data_final.drop(['DMSOLD'], axis = 1)
y = data_final.DMSOLD.values  
X = X.values

# scale the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
scaled_X = sc_X.fit_transform(X)
#scaled_X = pd.DataFrame(data = scaled_X, columns = X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
random_state = 0

def get_model_results(X_train, y_train, cat_cols):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('RF', RandomForestClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    initial_all_results = {}
    print("Running initial models")
    for name, model in models:
        print("Running initial "+name)
        kfold = StratifiedKFold(n_splits=10, random_state=1)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        
        results.append(cv_results)
        names.append(name)
        initial_all_results[name] = [cv_results.mean(),cv_results.std()]
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        classifier = model
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
    
    
#    # Compare Algorithms
#    pyplot.boxplot(results, labels=names)
#    pyplot.title('Algorithm Comparison')
#    pyplot.show()
    
    
    
    
    dataset_cols = cat_cols
    dataset_results = {}
    # Grid Search Logistic Regression
    print("Running LR Grid Search")
    LRclassifier = LogisticRegression(solver='liblinear', multi_class='ovr')
    LRpenalty = ['l1','l2']
    LRC=np.logspace(0, 4, 10)
    LRhyper = dict(C=LRC, penalty=LRpenalty)
    
    LRgrid_search = GridSearchCV(estimator = LRclassifier,
                               param_grid = LRhyper,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    LRgrid_search = LRgrid_search.fit(X_train, y_train)
    LRbest_accuracy = LRgrid_search.best_score_
    LRbest_parameters = LRgrid_search.best_params_
    dataset_results['LR'] = [LRbest_accuracy, LRbest_parameters]
    
    # Grid Search Linear Discriminant Analysis
    print("Running LDA Grid Search")
    LDAclassifier = LinearDiscriminantAnalysis()
    LDAparameters = [{'solver': ['svd']},
                  {'solver': ['lsqr','eigen'], 'shrinkage': ['auto']}]
    
    LDAgrid_search = GridSearchCV(estimator = LDAclassifier,
                               param_grid = LDAparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    LDAgrid_search = LDAgrid_search.fit(X_train, y_train)
    LDAbest_accuracy = LDAgrid_search.best_score_
    LDAbest_parameters = LDAgrid_search.best_params_
    dataset_results['LDA'] = [LDAbest_accuracy, LDAbest_parameters]
    
    # Grid Search KNN
    print("Running KNN Grid Search")
    KNNclassifier = KNeighborsClassifier()
    KNNparameters = [{'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'p':[1,2],'metric': ['minkowski']}]
    
    KNNgrid_search = GridSearchCV(estimator = KNNclassifier,
                               param_grid = KNNparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    KNNgrid_search = KNNgrid_search.fit(X_train, y_train)
    KNNbest_accuracy = KNNgrid_search.best_score_
    KNNbest_parameters = KNNgrid_search.best_params_
    dataset_results['KNN'] = [KNNbest_accuracy, KNNbest_parameters]
    dt.datetime.now()
    
    # Grid Search Decision Tree
    print("Running CART Grid Search")
    CARTclassifier = DecisionTreeClassifier()
    CARTparameters = [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}]
    
    CARTgrid_search = GridSearchCV(estimator = CARTclassifier,
                               param_grid = CARTparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    CARTgrid_search = CARTgrid_search.fit(X_train, y_train)
    CARTbest_accuracy = CARTgrid_search.best_score_
    CARTbest_parameters = CARTgrid_search.best_params_
    dataset_results['CART'] = [CARTbest_accuracy, CARTbest_parameters]
    
    # Grid Search SVC
    print("Running SVC Grid Search")
    SVCclassifier = SVC()
#    SVCparameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf','poly','sigmoid'],'degree':[3,4,5], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    SVCparameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    SVCgrid_search = GridSearchCV(estimator = SVCclassifier,
                               param_grid = SVCparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    SVCgrid_search = SVCgrid_search.fit(X_train, y_train)
    SVCbest_accuracy = SVCgrid_search.best_score_
    SVCbest_parameters = SVCgrid_search.best_params_
    dataset_results['SVC'] = [SVCbest_accuracy, SVCbest_parameters]
    
    
    # Grid Search Random Forest
    print("Running RF Grid Search")
    RFclassifier = RandomForestClassifier()
    RFparameters = [{'n_estimators': [10, 20, 40, 60], 'splitter': ['best', 'random']}]
    RFgrid_search = GridSearchCV(estimator = RFclassifier,
                               param_grid = RFparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    RFgrid_search = RFgrid_search.fit(X_train, y_train)
    RFbest_accuracy = RFgrid_search.best_score_
    RFbest_parameters = RFgrid_search.best_params_
    dataset_results['RF'] = [RFbest_accuracy, RFbest_parameters]
    print("Completed running all grid search")
    return dataset_results, dataset_cols



first_initial_all_results, first_dataset_results, first_dataset_cols = get_model_results(X_train, y_train, cat_cols)






# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()