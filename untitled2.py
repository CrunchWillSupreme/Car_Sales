# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:55:30 2019
@author: willh
"""
### import packages
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
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE
import datetime as dt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.linalg.decomp_svd import LinAlgError


### functions
def get_data():
    """ Getting the data from either my laptop or computer, and add a couple date-related columns """
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
    # Make Date-related columns
    raw_data['DMSTDESL_DT'] = pd.to_datetime(raw_data['DMSTDESL'], format = '%Y%m%d')
    raw_data['Sale_day_name'] = raw_data['DMSTDESL_DT'].dt.weekday
    # make a column for week of month (1st week, 2nd week, 3rd week, ...)
    days = pd.DatetimeIndex(raw_data['DMSTDESL_DT'].unique())
    week_month = {}
    week = 1
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
            
    raw_data['month_week'] = raw_data['DMSTDESL_DT'].map(week_month)
    season = pd.DataFrame({'M':[1,2,3,4,5,6,7,8,9,10,11,12],'Season':['W','W','S','S','S','Su','Su','Su','F','F','F','W']})
    raw_data = pd.merge(raw_data,season,left_on='DMMONTH',right_on='M')
    raw_data.drop('M',inplace=True,axis=1)
    raw_data['STIMES_bin2'].replace({"<10":"before 10", ">2":"after 2"}, inplace= True)
    raw_data['Arbitration_bins'].replace({"<=10":"less than 10", ">150":"greater than 150"}, inplace= True)
    raw_data = raw_data[raw_data['DMMODEL'].notnull()]
    raw_data = raw_data[raw_data['DMJDCAT'].notnull()]
    return data_dict, raw_data, case_study

def prep_data(data_penult, cat_cols = False, scaling = False):
    """ Prep data - make dummy variables, scale the data, split data into train and test sets """
    if cat_cols:
        # Make dummy variables
#        cat_cols = data_penult.loc[:, data_penult.columns != 'DMSOLD'].columns # use this if all columns are categorical
#        cat_cols = ['STIMESxArb', 'month_week'] # use this if columns are both categorical and continuous
        for col in cat_cols:
            cat_list='var'+'_'+col
            cat_list = pd.get_dummies(data_penult[col], prefix=col, drop_first = True)
            data1=data_penult.join(cat_list)
            data_penult=data1
        data_vars=data_penult.columns.values.tolist()
        to_keep=[i for i in data_vars if i not in cat_cols]
        # final data columns will be:
        data_final=data_penult[to_keep]
#        data_final.columns.values
    else:
        data_final = data_penult
    # Make X and y datasets
    X = data_final.drop(['DMSOLD'], axis = 1)
    y = data_final.DMSOLD 
    #X = X.values
    if scaling:
        # scale the data - If dataset has continuous variables
        x_index = X.index
        x_cols = X.columns        
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        X = pd.DataFrame(data=X, index = x_index, columns=x_cols)
        #scaled_X = pd.DataFrame(data = scaled_X, columns = X.columns.tolist())        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    return X_train, X_test, y_train, y_test
                                                          
def get_model_results(X_train, y_train):
    """ Run data through models to see scores, confusion matrices, and important features """
    # Spot Check Algorithms
    # evaluate each model in turn
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #    models.append(('SVM', SVC(gamma='auto')))
    models.append(('XGB', XGBClassifier()))
    models.append(('RF', RandomForestClassifier()))

    results = []
    names = []
    initial_all_results = {}
    columns = {}
    cm_dict = {}
    scores_dict = {}
    feature_importances = {}
    prec_rec = {}
    print("Running initial models")
    for name, model in models:
        try:
            timenow = dt.datetime.now()
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
            cm_dict[name] = cm
            score = classifier.score(X_test, y_test)
            print("score:" + str(score))
            scores_dict[name] = score
            prec_score = precision_score(y_test,y_pred)
            rec_score = recall_score(y_test, y_pred)
            prec_rec[name] = [prec_score, rec_score]
            print("prec_rec: ", prec_score, rec_score)
            if name not in ['CART','RF','NB','XGB']:
                coef = classifier.coef_
                abs_coef = abs(coef)
                sorted_index = np.argsort(abs_coef)
        #        sorted_coef = abs_coef[sorted_index]
                sorted_cols = X_test.columns[sorted_index]
                columns[name] = sorted_cols
            elif name in ['CART','NB']:
                continue
            else:
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                feature_importances[name] = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        except LinAlgError:
            print("LinAlgError: SVD did not converge")
#        coef_dict = {k:v for (k,v) in zip(sorted_cols, sorted_coef)}
#        print(sorted_cols)
#        coefs = pd.DataFrame(coef_dict)
#        acc_score = classifier.accuracy_score(y_test, y_pred)
#        print("acc_score:" + str(acc_score))
#    # Compare Algorithms
#    pyplot.boxplot(results, labels=names)
#    pyplot.title('Algorithm Comparison')
#    pyplot.show()
    
    return initial_all_results, cm_dict, scores_dict, columns, feature_importances, prec_rec

def rfe_feature_selection(X_train, y_train):
    models = []
#    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
#    models.append(('NB', GaussianNB()))
    #    models.append(('SVM', SVC(gamma='auto')))
#    models.append(('XGB', XGBClassifier()))
    models.append(('RF', RandomForestClassifier()))
    rfe_columns = {}
    for name, model in models: 
        rfe = RFE(model, 10)
        rfe = rfe.fit(X_train, y_train)
        print(rfe.support_)
        print(rfe.ranking_)
    
        indices = [i for i, x in enumerate(rfe.ranking_) if x == 1]
        cols = X_train.iloc[:,indices].columns
        rfe_columns[model] = cols
    return rfe_columns
            
def grid_search(X_train, y_train):    
    """ Run grid searches on models to find the best hyperparameters """
    gridsearch_results = {}
#    # Grid Search Logistic Regression
#    print("Running LR Grid Search")
#    LRclassifier = LogisticRegression(solver='liblinear', multi_class='ovr')
#    LRpenalty = ['l1','l2']
#    LRC=np.logspace(0, 4, 10)
#    LRhyper = dict(C=LRC, penalty=LRpenalty)
#    
#    LRgrid_search = GridSearchCV(estimator = LRclassifier,
#                               param_grid = LRhyper,
#                               scoring = 'accuracy',
#                               cv = 10,
#                               n_jobs = -1)
#    LRgrid_search = LRgrid_search.fit(X_train, y_train)
#    LRbest_accuracy = LRgrid_search.best_score_
#    LRbest_parameters = LRgrid_search.best_params_
#    gridsearch_results['LR'] = [LRbest_accuracy, LRbest_parameters]
#    
#    # Grid Search Linear Discriminant Analysis
#    print("Running LDA Grid Search")
#    LDAclassifier = LinearDiscriminantAnalysis()
#    LDAparameters = [{'solver': ['svd']},
#                  {'solver': ['lsqr','eigen'], 'shrinkage': ['auto']}]
#    
#    LDAgrid_search = GridSearchCV(estimator = LDAclassifier,
#                               param_grid = LDAparameters,
#                               scoring = 'accuracy',
#                               cv = 10,
#                               n_jobs = -1)
#    LDAgrid_search = LDAgrid_search.fit(X_train, y_train)
#    LDAbest_accuracy = LDAgrid_search.best_score_
#    LDAbest_parameters = LDAgrid_search.best_params_
#    gridsearch_results['LDA'] = [LDAbest_accuracy, LDAbest_parameters]
#    
#    # Grid Search KNN
#    print("Running KNN Grid Search")
#    KNNclassifier = KNeighborsClassifier()
#    KNNparameters = [{'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'p':[1,2],'metric': ['minkowski']}]
#    
#    KNNgrid_search = GridSearchCV(estimator = KNNclassifier,
#                               param_grid = KNNparameters,
#                               scoring = 'accuracy',
#                               cv = 10,
#                               n_jobs = -1)
#    KNNgrid_search = KNNgrid_search.fit(X_train, y_train)
#    KNNbest_accuracy = KNNgrid_search.best_score_
#    KNNbest_parameters = KNNgrid_search.best_params_
#    gridsearch_results['KNN'] = [KNNbest_accuracy, KNNbest_parameters]
#    dt.datetime.now()
    
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
    gridsearch_results['CART'] = [CARTbest_accuracy, CARTbest_parameters]
    
#    # Grid Search SVC
#    print("Running SVC Grid Search")
#    SVCclassifier = SVC()
#    #    SVCparameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#    #                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf','poly','sigmoid'],'degree':[3,4,5], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#    SVCparameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#    SVCgrid_search = GridSearchCV(estimator = SVCclassifier,
#                               param_grid = SVCparameters,
#                               scoring = 'accuracy',
#                               cv = 10,
#                               n_jobs = -1)
#    SVCgrid_search = SVCgrid_search.fit(X_train, y_train)
#    SVCbest_accuracy = SVCgrid_search.best_score_
#    SVCbest_parameters = SVCgrid_search.best_params_
#    gridsearch_results['SVC'] = [SVCbest_accuracy, SVCbest_parameters]
    
    # Grid Search XGBoost
    print("Running XGB Grid Search")
    XGBclassifier = XGBClassifier()
    XGBparameters = [{'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.03, 0.05, 0.1]}]
    XGBgrid_search = GridSearchCV(estimator = XGBclassifier,
                               param_grid = XGBparameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    XGBgrid_search = XGBgrid_search.fit(X_train, y_train)
    XGBbest_accuracy = XGBgrid_search.best_score_
    XGBbest_parameters = XGBgrid_search.best_params_
    gridsearch_results['XGB'] = [XGBbest_accuracy, XGBbest_parameters]
    print("Completed running all grid search")
    
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
    gridsearch_results['RF'] = [RFbest_accuracy, RFbest_parameters]
    print("Completed running all grid search")
    
    return gridsearch_results


data_dict, raw_data, case_study = get_data()        

# Make copy of data
data = raw_data.copy()

 

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

## Drop columns that I won't be using
#drop_cols = ['SFLOOR', 'DMECRDATE','DMSTDESL', 'DMSALWK', 'SSALE_', 'SRUN_', 'STIMES', 'DMTRANTYPE', 'DMSELLRNM', 'DMJDCAT', 
#             'DMMODEL', 'DMBODY', 'SSER17', 'AGEDDAYS', 'DMPRECOND', 'DMPOSTCOND', 'DMDETFEE', 'DMRECONFEE','SFLR_lower','% Arbitration','imp_STIMES','VNMMR','SSLEPR'
#             , 'DMOPCSUID','DMMODELYR','SMILES','DMMONTH','SLANE_','STIMES_bin3','STIMES_bin', 'DMSTDESL_DT','Sale_day_name']
#data_penult = data.drop(drop_cols, axis =1)
#data[data.columns != drop_cols]

### feature exploration - AGEDDAYS contributes most
data_penult = data[['DMSOLD', 'AGEDDAYS']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
AGEDDAYS_results, AGEDDAYS_cm, AGEDDAYS_score, AGEDDAYS_columns, AGEDDAYS_feature_importances, AGEDDAYS_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'AGEDDAYS','DMDETFEE']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
AGEDFEE_results, AGEDFEE_cm, AGEDFEE_score, AGEDFEE_columns, AGEDFEE_feature_importances, AGEDFEE_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'AGEDDAYS','imp_STIMES']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
AGEDimpS_results, AGEDimpS_cm, AGEDimpS_score, AGEDimpS_columns, AGEDimpS_feature_importances, prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD','% Arbitration', 'AGEDDAYS','DMECRDATE_IND']]
cat_cols = ['DMECRDATE_IND']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
ARBAGED_results, ARBAGED_cm, ARBAGED_score, ARBAGED_columns, ARBAGED_feature_importances, ARBAGED_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD','% Arbitration', 'AGEDDAYS']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
ARB_results, ARB_cm, ARB_score, ARB_columns, ARB_feature_importances, ARB_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD','% Arbitration', 'AGEDDAYS','STIMES_bin3', 'DMSELLRNM','DMJDCAT']]
cat_cols = ['STIMES_bin3', 'DMSELLRNM','STIMES_bin3','DMJDCAT']
X_train, X_test, y_train, y_test = prep_data(data_penult,cat_cols)
ARB_results, ARB_cm, ARB_score, ARB_columns, ARB_feature_importances, ARB_prec_rec = get_model_results(X_train, y_train)

### Individual features ##################################################
data_penult = data[['AGEDDAYS','SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3', '% Arbitration', 'DMMODELYR','DMSOLD','DMSELLRNM','DMJDCAT','month_week','Season']]
cat_cols = ['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3','DMSELLRNM','DMJDCAT','month_week','Season']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
all_results, all_cm, all_score, all_columns, all_feature_importances, all_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['AGEDDAYS','SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3', '% Arbitration', 'DMMODELYR','DMSOLD','DMSELLRNM','DMJDCAT','month_week','Season']]
cat_cols = ['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3','DMSELLRNM','DMJDCAT','month_week','Season']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols, scaling = True)
scaledall_results, scaledall_cm, scaledall_score, scaledall_columns, scaledall_feature_importances, scaledall_prec_rec = get_model_results(X_train, y_train)

rfe_columns = rfe_feature_selection(X_train, y_train)

data_penult = data[['AGEDDAYS','SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3', '% Arbitration', 'DMMODELYR','DMSOLD']]
cat_cols = ['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin3']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_11, cm_11, score_11, columns_11, feature_importances_11, prec_rec = get_model_results(X_train, y_train)

data_penult = data[['AGEDDAYS','DMECRDATE_IND', '% Arbitration', 'DMMODELYR','STIMES_bin3','DMSOLD']]
cat_cols = ['DMECRDATE_IND', 'STIMES_bin3']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_21, cm_21, score_21, columns_21, feature_importances_21, prec_rec_21 = get_model_results(X_train, y_train)

data_penult = data[['AGEDDAYS','DMECRDATE_IND', '% Arbitration', 'DMMODELYR','DMSOLD']]
cat_cols = ['DMECRDATE_IND']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_22, cm_22, score_22, columns_22, feature_importances_22, prec_rec_22 = get_model_results(X_train, y_train)

data_penult = data[['AGEDDAYS','DMECRDATE_IND', '% Arbitration', 'DMMODELYR','Season','month_week', 'DMJDCAT','DMSOLD']]
cat_cols = ['DMECRDATE_IND', 'Season', 'month_week','DMJDCAT']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_31, cm_31, score_31, columns_31, feature_importances_31, prec_rec_31 = get_model_results(X_train, y_train)

data_penult = data[['AGEDDAYS','DMECRDATE_IND', '% Arbitration', 'DMMODELYR','Season','month_week','DMSOLD']]
cat_cols = ['DMECRDATE_IND', 'Season', 'month_week']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_32, cm_32, score_32, columns_32, feature_importances_32, prec_rec_32 = get_model_results(X_train, y_train)



data_penult = data[['DMSOLD', 'DMSELLRNM']]
cat_cols = ['DMSELLRNM']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
slronly_results, slronly_cm, slronly_score, slronly_columns, slronly_feature_importances, slronly_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['STIMES_bin3','DMSOLD']]
cat_cols = ['STIMES_bin3']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
STIMES3_results, STIMES3_cm, STIMES3_score, STIMES3_columns, STIMES3_feature_importances, STIMES3_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'DMMODEL']]
cat_cols = ['DMMODEL']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
modelonly_results, modelonly_cm, modelonly_score, modelonly_columns, modelonly_feature_importances, modelonly_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', '% Arbitration']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
ARBonly_results, ARBonly_cm, ARBonly_score, ARBonly_columns, ARBonly_feature_importances, ARBonly_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'DMJDCAT']]
cat_cols = ['DMJDCAT']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
DMJDCAT_results, DMJDCAT_cm, DMJDCAT_score, DMJDCAT_columns, DMJDCAT_feature_importances, DMJDCAT_prec_rec = get_model_results(X_train, y_train)

#data_penult = data[['DMSOLD', 'DMJDCAT']]
#cat_cols = ['DMJDCAT']
#X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
#X_train.drop(columns=['DMJDCAT_FULLSIZE CAR'])
#X_test.drop(columns=['DMJDCAT_FULLSIZE CAR'])
#DMJDCAT_results_1, DMJDCAT_cm_1, DMJDCAT_score_1, DMJDCAT_columns_1, DMJDCAT_feature_importances_1, DMJDCAT_prec_rec_1 = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'month_week']]
cat_cols = ['month_week']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
mWEEK_results, mWEEK_cm, mWEEK_score, mWEEK_columns, mWEEK_feature_importances, mWEEK_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'Season']]
cat_cols = ['Season']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
season_results, season_cm, season_score, season_columns, season_feature_importances, season_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMMODELYR', 'DMSOLD']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
MODELYR_results, MODELYR_cm, MODELYR_score, MODELYR_columns, MODELYR_feature_importances, MODELYR_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'DMECRDATE_IND']]
cat_cols = ['DMECRDATE_IND']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
DMECR_results, DMECR_cm, DMECR_score, DMECR_columns, DMECR_feature_importances, DMECR_prec_rec = get_model_results(X_train, y_train)

# svd could not converge error
data_penult = data[['DMSOLD', 'DMSELLRNM','DMMODEL']] #slr
cat_cols = ['DMSELLRNM','DMMODEL']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
slr_results, slr_cm, slr_score, slr_columns, slr_feature_importances, slr_prec_rec = get_model_results(X_train, y_train)


data_penult = data[['DMSOLD', 'SMILES']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
SMILES_results, SMILES_cm, SMILES_score, SMILES_columns, SMILES_feature_importances, SMILES_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['imp_STIMES','DMSOLD']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
impSTIMES_results, impSTIMES_cm, impSTIMES_score, impSTIMES_columns, impSTIMES_feature_importances, impSTIMES_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['STIMES_bin2','DMSOLD']]
cat_cols = ['STIMES_bin2']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
STIMES2_results, STIMES2_cm, STIMES2_score, STIMES2_columns, STIMES2_feature_importances, STIMES2_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'DMMAKE']]
cat_cols = ['DMMAKE']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
MAKE_results, MAKE_cm, MAKE_score, MAKE_columns, MAKE_feature_importances, MAKE_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMSOLD', 'SFLOOR_IND']]
cat_cols = ['SFLOOR_IND']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
FLRind_results, FLRind_cm, FLRind_score, FLRind_columns, FLRind_feature_importances, FLRind_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMMODELYR', 'DMSOLD']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
DMMODELYR_results, DMMODELYR_cm, DMMODELYR_score, DMMODELYR_columns, DMMODELYR_feature_importances, DMMODELYR_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['VNMMR', 'DMSOLD']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
VNMMR_results, VNMMR_cm, VNMMR_score, VNMMR_columns, VNMMR_feature_importances, VNMMR_prec_rec = get_model_results(X_train, y_train)
### interactions ################################################################################################

data['ARBxMDLYR'] = data['Arbitration_bins'] + ' ' + data['DMMODELYR'].map(str)
data_penult = data[['ARBxMDLYR', 'DMSOLD']]
cat_cols = ['ARBxMDLYR']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
ARBxMDLR_results, ARBxMDLR_cm, ARBxMDLR_score, ARBxMDLR_columns, ARBxMDLR_feature_importances, ARBxMDLR_prec_rec = get_model_results(X_train, y_train)

data_penult = data[['DMMODELYR','Arbitration_bins', 'DMSOLD']]
cat_cols = ['Arbitration_bins']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
ARBMDLR_results, ARBMDLR_cm, ARBMDLR_score, ARBMDLR_columns, ARBMDLR_feature_importances, ARBMDLR_prec_rec = get_model_results(X_train, y_train)


data['CATxMAKE'] = data['DMJDCAT'] + ' ' + data['DMMAKE']
data_penult = data[['DMSOLD', 'CATxMAKE']]
cat_cols = ['CATxMAKE']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
CATMAKE_results, CATMAKE_cm, CATMAKE_score, CATMAKE_columns, CATMAKE_feature_importances, CATMAKE_prec_rec = get_model_results(X_train, y_train)


data['SLRxSTIMES'] = data['DMSELLRNM'] + ' ' + data['STIMES_bin3']
data_penult = data[['DMSOLD', 'SLRxSTIMES']]
cat_cols = ['SLRxSTIMES']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
SLRxTIMES_results, SLRxTIMES_cm, SLRxTIMES_score, SLRxTIMES_columns, SLRxTIMES_feature_importances, SLRxTIMES_prec_rec = get_model_results(X_train, y_train)


data['SFLRxVNMMR'] = data['SFLOOR_IND'] + ' ' + data['VNMMR'].map(str)
data_penult = data[['DMSOLD', 'SFLRxVNMMR']]
cat_cols = ['SFLRxVNMMR']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
SLRTIMES_results, SLRTIMES_cm, SLRTIMES_score, SLRTIMES_columns, SLRTIMES_feature_importances, SLRTIMES_prec_rec = get_model_results(X_train, y_train)


data['AGEDxECR'] = data['AGEDDAYS'].map(str) + ' ' + data['DMECRDATE_IND']
data_penult = data[['DMSOLD', 'AGEDxECR']]
cat_cols = ['AGEDxECR']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
AGEECR_results, AGEECR_cm, AGEECR_score, AGEECR_columns, AGEECR_feature_importances, AGEECR_prec_rec = get_model_results(X_train, y_train)

data['AGEDxSLR'] = data['AGEDDAYS'].map(str) + ' ' + data['DMSELLRNM']
data_penult = data[['DMSOLD', 'AGEDxSLR']]
cat_cols = ['AGEDxSLR']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
AGESLR_results, AGESLR_cm, AGESLR_score, AGESLR_columns, AGESLR_feature_importances, AGESLR_prec_rec = get_model_results(X_train, y_train)





data_penult = data[['DMSTDESL', 'DMMONTH', 'DMSALWK', 'SLANE_', 'SRUN_', 'imp_STIMES','DMSOLD', 'DMSELLRNM',
                    'DMMODELYR', 'DMMODEL','SMILES', 'DMJDCAT', 'VNMMR', '% Arbitration','SFLOOR_IND','DMECRDATE_IND']]
cat_cols = ['DMSELLRNM','DMMODEL','DMJDCAT','SFLOOR_IND','DMECRDATE_IND']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)        
results, cm, score, columns, feature_importances = get_model_results(X_train, y_train)


data_penult = data[['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins','DMSOLD']]
cat_cols = ['SFLOOR_IND','DMECRDATE_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
results_1, cm_1, score_1, columns_1, feature_importances_1 = get_model_results(X_train, y_train)


data_penult = data[['DMSOLD', 'DMSELLRNM','DMMODEL','imp_STIMES']]
cat_cols = ['DMSELLRNM','DMMODEL']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols, scaling = True)
slr2_results, slr2_cm, slr2_score, slr2_columns, slr2_feature_importances = get_model_results(X_train, y_train)


data_penult = data[['DMSOLD', 'AGEDDAYS','DMSELLRNM','DMMODEL']]
cat_cols = ['DMSELLRNM','DMMODEL']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
AGEDSLR_results, AGEDSLR_cm, AGEDSLR_score, AGEDSLR_columns, AGEDSLR_feature_importances, AGEDSLR_prec_rec = get_model_results(X_train, y_train)

# gives error on LDA - SVD did not converge
data['SLRxMODEL'] = data['DMSELLRNM'] + ' ' + data['DMMODEL']
data = data[data['DMMODEL'].notnull()]
data['SLRxMODEL'].unique()
data_penult = data[['SLRxMODEL', 'DMSOLD']]
cat_cols = ['SLRxMODEL']
X_train, X_test, y_train, y_test = prep_data(data_penult, cat_cols)
SLRxMODEL_results, SLRxMODEL_cm, SLRxMODEL_score, SLRxMODEL_columns, SLRxMODEL_feature_importances = get_model_results(X_train, y_train)

    


data_penult = data[['SFLOOR_IND', 'STIMES_bin2', 'Arbitration_bins', 'DMMODELYR_bins','DMSOLD']]
cat_cols = data_penult.loc[:, data_penult.columns != 'DMSOLD'].columns
X_train, X_test, y_train, y_test = prep_data(data_penult)
MODELYR_results, MODELYR_cm, MODELYR_score, MODELYR_columns, MODELYR_feature_importances = get_model_results(X_train, y_train) 


data_penult = data[['DMDETFEE','DMSOLD']]
X_train, X_test, y_train, y_test = prep_data(data_penult)
DETFEE_results, DETFEE_cm, DETFEE_score, DETFEE_columns, DETFEE_feature_importances, prec_rec = get_model_results(X_train, y_train) 


data['STIMESxArb'] = data['STIMES_bin2'] + " " + data['Arbitration_bins']
ECRdata = data[data['SFLOOR_IND'] == 'N']
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




rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
rfe.feature_importances_
pd.DataFrame({"features":X_train.columns, "importance":rfe.feature_importances_})


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()  


