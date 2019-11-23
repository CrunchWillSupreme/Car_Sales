# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:47:30 2019

@author: Will Han
https://www.predictiveanalyticsworld.com/patimes/on-variable-importance-in-logistic-regression/9649/
https://towardsdatascience.com/model-based-feature-importance-d4f6fb2ad403
https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model
https://github.com/vishu160196/Feature-importance/blob/master/Feature%20importance%20using%20model%20parameters%20(LR).ipynb
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


os.chdir(r'C:\Codes\Cox_Manheim_analysis')
data_dict = pd.read_excel('Case Study and Data Dictionary 201407.xlsx', sheet_name = 1, skiprows = 1)
#raw_data = pd.read_csv('Case_Study_Transaction_Data_201407_modified.csv')
raw_data = pd.read_excel('transaction_data_processed.xlsx', sheet_name = 'data')
with open(r'C:\Codes\Cox_Manheim_analysis\Case_Study.txt') as f:
    case_study = f.read()
#print(case_study)

c = raw_data.columns

# Take a look at some comparisons
pd.crosstab(data['DMMONTH'], data['DMSOLD']).plot(kind='bar')
pd.crosstab(data['DMMAKE'], data['DMSOLD']).plot(kind='bar')
pd.crosstab(data['DMJDCAT'], data['DMSOLD']).plot(kind='bar')
pd.crosstab(data['SLANE_'], data['DMSOLD']).plot(kind='bar')

sold = raw_data[raw_data['DMSOLD'] == 'Y']
not_sold = raw_data[raw_data['DMSOLD'] == 'N']
sns.distplot(not_sold['SFLOOR'])
sns.distplot(sold['SFLOOR'])

# Make copy of data
data = raw_data.copy()
data.sort_values(by=['DMSTDESL', 'SLANE_', 'SRUN_'], inplace = True)


## Convert 0 to NaN
#zeros_conv = ['SFLOOR', 'DMECRDATE', 'VNMMR']
#for col in zeros_conv:
#    data[col] = data[col].replace([0], np.NaN)

# Make categorical variables
cat_cols = ['DMMONTH', 'SLANE_', 'DMOPCSUID', 'DMMAKE','STIMES_bin', 'Arbitration_bins','DMMODELYR','DMMODELYR_bins']
for col in cat_cols:
    data[col] = data[col].astype('category')

## Make indicator columns (Y/N)     
#data['SFLOOR_IND'] = np.where(np.isnan(data['SFLOOR']), 'N', 'Y')
#data['DMECRDATE_IND'] = np.where(np.isnan(data['DMECRDATE']), 'N', 'Y')
#data['DMSOLD'] = np.where(data['DMSOLD'] == 'N', 0, 1)
    
## Format column to float
#data['% Arbitration'] = data['% Arbitration'].str.replace('%', '').astype(float)

# Drop columns that I won't be using
drop_cols = ['SFLOOR', 'DMECRDATE','DMSTDESL', 'DMSALWK', 'SSALE_', 'SRUN_', 'STIMES', 'DMTRANTYPE', 'DMSELLRNM', 'DMJDCAT', 'DMMODEL', 'DMBODY', 'SSER17', 'AGEDDAYS', 'DMPRECOND', 'DMPOSTCOND', 'DMDETFEE', 'DMRECONFEE','SFLR_lower','% Arbitration','imp_STIMES','VNMMR','SSLEPR'
             ,'DMMAKE', 'DMOPCSUID','DMMODELYR','SMILES','DMMONTH','SLANE_']
data_penult = data.drop(drop_cols, axis =1)

# Make dummy variables
cat_cols = ['DMECRDATE_IND','SFLOOR_IND','STIMES_bin', 'Arbitration_bins','DMMODELYR_bins']
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

data_final.isnull().sum()

## only 162 missing values for % Arbitration, ~0.3%.  Lets remove these
#data_final['% Arbitration'].isnull().sum()/len(data_final['% Arbitration'])
#data_final = data_final[data_final['% Arbitration'].notnull()]
#
## 4168 missing values for VNMMR, ~7.7%.  Remove these too
#data_final['VNMMR'].isnull().sum()/len(data_final['VNMMR'])
#data_final = data_final[data_final['VNMMR'].notnull()]


# Checking proportion of dependent variable
# packages that help balance proportions: sklearn.model_selection.StratifiedShuffleSplit, from imblearn.over_sampling import SMOTE
data_final['SFLOOR_IND_Y'].value_counts().plot.bar()
data_final['DMSOLD'].value_counts().plot.bar()


# Make X and y datasets
X = data_final.drop(['DMSOLD'], axis = 1)
y = data_final.DMSOLD  

data_final.dtypes


# Inspect dependent variable - DMSOLD    
data['DMSOLD'].value_counts()

sns.countplot(x='DMSOLD', data = data, palette = 'hls')
plt.show()
plt.savefig('count_plot')


count_no_purchase = len(data[data['DMSOLD'] == 0])
count_purchase = len(data[data['DMSOLD'] == 1])
pct_of_no_purchase = count_no_purchase/(count_no_purchase + count_purchase)
print("percentage of no purchase is", pct_of_no_purchase*100)
pct_of_purchase = count_purchase/(count_no_purchase + count_purchase)
print("percentage of purchase is", pct_of_purchase*100)

# Scale the data set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
scaled_X = sc_X.fit_transform(X)
scaled_X = pd.DataFrame(data = scaled_X, columns = X.columns.tolist())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.3, random_state = 0)
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
classifier.score(X_train, y_train)
classifier.score(X_test, y_test)
rfe2 = RFE(classifier, 6)
rfe2 = rfe2.fit(X_train, y_train)
print(rfe2.support_)
print(rfe2.ranking_)
indices2 = [i for i, x in enumerate(rfe2.ranking_) if x == 1]
cols2 = X_train.iloc[:,indices2].columns

# RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE

# Logistic Regression - Feature Importance #1
logreg = LogisticRegression()

rfe = RFE(logreg, 6)
rfe = rfe.fit(scaled_X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

indices = [i for i, x in enumerate(rfe.ranking_) if x == 1]
cols = X.iloc[:,indices].columns
scaled_X_2 = scaled_X[cols]
y = y['DMSOLD']

import statsmodels.api as sm
logit_model = sm.Logit(y,scaled_X)
result = logit_model.fit()
print(result.summary2())
# remove the variables where p value is insignificant
cols=[]
#X=os_data_X[cols]
#y=os_data_y['y']
logit_model=sm.Logit(y,scaled_X_2)
result2=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(scaled_X_2, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# Logistic Regression - Feature Importance #2
clf = LogisticRegression()
clf.fit(scaled_X,y)

feature_importance = abs(clf.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()   
plt.show()


# Logistic Regression - Feature Importance #3
clf2 = LogisticRegression()
clf2.fit(scaled_X,y)

clf2.coef_.T
np.array([scaled_X.columns[:-1]]).T

feature_importance=pd.DataFrame(np.hstack((np.array([scaled_X.columns]).T, clf2.coef_.T)), columns=['feature', 'importance'])
feature_importance['importance']=pd.to_numeric(abs(feature_importance['importance']))
feature_importance.sort_values(by='importance', ascending=False)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train.values, y_train.values)
y_pred = classifier.predict(X_test.values)
cm = confusion_matrix(y_test.values.tolist(), y_pred.tolist())
# RF - Feature Importance 1
# Unscaled X
seed = 0
rf = RandomForestClassifier(n_estimators = 100, random_state = seed)

rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame({"Feature": scaled_X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(16,8))
sns.barplot(x="Feature", y="Importance", data=feature_importances)
plt.title("Feature Importances for Random Forest Model")
plt.xticks(rotation="vertical")
plt.show()

# Scaled X
seed = 0
rf = RandomForestClassifier(n_estimators = 100, random_state = seed)

rf.fit(scaled_X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame({"Feature": scaled_X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(16,8))
sns.barplot(x="Feature", y="Importance", data=feature_importances)
plt.title("Feature Importances for Random Forest Model")
plt.xticks(rotation="vertical")
plt.show()

rfclassifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rfclassifier.fit(scaled_X, y)
importances = rfclassifier.feature_importances_
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame({"Feature": scaled_X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(16,8))
sns.barplot(x="Feature", y="Importance", data=feature_importances)
plt.title("Feature Importances for Random Forest Model")
plt.xticks(rotation="vertical")
plt.show()


# RF - Feature Importance 2
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
n_list=[3,5,7,9,11,13,17,29,33,47,51,101,203,501,1001]
cv_err=[]
train_err=[]
for n in n_list:
    clf=RandomForestClassifier(n_estimators=n, class_weight='balanced', n_jobs=-1)
    clf.fit(X, y)
    sig_clf=CalibratedClassifierCV(clf)
    sig_clf.fit(X, y)
    
#    predict_y=sig_clf.predict_proba(x_test)
#    cv_err.append(log_loss(y_test, predict_y))
#    
#    predict_y=sig_clf.predict_proba(x_train)
#    train_err.append(log_loss(y_train, predict_y))
#
#plt.plot(n_list, cv_err, label='cv error', c='b')
#plt.plot(n_list, train_err, label='train error', c='r')
#plt.legend()
#plt.show()