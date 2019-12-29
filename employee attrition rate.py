# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:39:34 2019

@author: reddymv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel("MFG10year_Data.xlsx")
data.info()
data.shape
data.describe()
data.isna().sum()
data.info()
data = data.drop(['birthdate_key', 'recorddate_key', 'orighiredate_key', 'terminationdate_key', 
                          'termreason_desc', 'termtype_desc', 'department_name', 'gender_full'], axis=1)
#counting the values of all columns having multi values to convert to integers
data['EmployeeID'].value_counts()
data['city_name'].value_counts()
data['job_title'].value_counts()
data['store_name'].value_counts()
data['gender_short'].value_counts()
data['STATUS_YEAR'].value_counts()
data['BUSINESS_UNIT'].value_counts()
#storing job titles in to categories
board = ['VP Stores', 'Director, Recruitment', 'VP Human Resources', 'VP Finance',
         'Director, Accounts Receivable', 'Director, Accounting',
         'Director, Employee Records', 'Director, Accounts Payable',
         'Director, HR Technology', 'Director, Investments',
         'Director, Labor Relations', 'Director, Audit', 'Director, Training',
         'Director, Compensation']

executive = ['Exec Assistant, Finance', 'Exec Assistant, Legal Counsel',
             'CHief Information Officer', 'CEO', 'Exec Assistant, Human Resources',
             'Exec Assistant, VP Stores']

manager = ['Customer Service Manager', 'Processed Foods Manager', 'Meats Manager',
           'Bakery Manager', 'Produce Manager', 'Store Manager', 'Trainer', 'Dairy Manager']


employee = ['Meat Cutter', 'Dairy Person', 'Produce Clerk', 'Baker', 'Cashier',
            'Shelf Stocker', 'Recruiter', 'HRIS Analyst', 'Accounting Clerk',
            'Benefits Admin', 'Labor Relations Analyst', 'Accounts Receiveable Clerk',
            'Accounts Payable Clerk', 'Auditor', 'Compensation Analyst',
            'Investment Analyst', 'Systems Analyst', 'Corporate Lawyer', 'Legal Counsel']
#custom function to change all values to store categories
def changeTitle(row):
    if row in board:
        return 'board'
    elif row in executive:
        return 'executive'
    elif row in manager:
        return 'manager'
    else:
        return 'employee'
#To change column name
data['job_title'] = data['job_title'].apply(changeTitle)
#assigning numbers to the values
data['job_title'] = data['job_title'].map({'board': 3, 'executive': 2, 'manager': 1, 'employee': 0})
city_pop_2011 = {'Vancouver':2313328,
                 'Victoria':344615,
                 'Nanaimo':146574,
                 'New Westminster':65976,
                 'Kelowna':179839,
                 'Burnaby':223218,
                 'Kamloops':85678,
                 'Prince George':71974,
                 'Cranbrook':19319,
                 'Surrey':468251,
                 'Richmond':190473,
                 'Terrace':11486,
                 'Chilliwack':77936,
                 'Trail':7681,
                 'Langley':25081,
                 'Vernon':38180,
                 'Squamish':17479,
                 'Quesnel':10007,
                 'Abbotsford':133497,
                 'North Vancouver':48196,
                 'Fort St John':18609,
                 'Williams Lake':10832,
                 'West Vancouver':42694,
                 'Port Coquitlam':55985,
                 'Aldergrove':12083,
                 'Fort Nelson':3561,
                 'Nelson':10230,
                 'New Westminister':65976,
                 'Grand Forks':3985,
                 'White Rock':19339,
                 'Haney':76052,
                 'Princeton':2724,
                 'Dawson Creek':11583,
                 'Bella Bella':1095,
                 'Ocean Falls':129,
                 'Pitt Meadows':17736,
                 'Cortes Island':1007,
                 'Valemount':1020,
                 'Dease Lake':58,
                 'Blue River':215}



data['population'] = data['city_name']
data['population'] = data.population.map(city_pop_2011)
data['population_category'] = data.population

# categorize based on the population size

city_ix = (data['population'] >= 100000)
rural_ix = ((data['population'] < 100000) & (data['population'] >= 10000))
remote_ix = (data['population'] < 10000)
data.loc[city_ix, 'population_category'] = 'City'
data.loc[rural_ix, 'population_category'] = 'Rural'
data.loc[remote_ix, 'population_category'] = 'Remote'
data.population_category.value_counts()
data['gender_short']=data['gender_short'].map({'M':'1','F':'0'})
data['status'] = data['status'].map({'ACTIVE': 1, 'TERMINATED': 0})
data['BUSINESS_UNIT'] = data['BUSINESS_UNIT'].map({'STORES': 0, 'HEADOFFICE' :1})
out_of_co = data[data.status == 0]
in_co = data[data.status == 1]
import seaborn as sns
data = data.drop(columns = ['length of service','Pop'])
data['population_category'].value_counts()
data['population_category']=data['population_category'].map({'City':'0', 'Rural':'1','Remote':'2'})
Y=data['status']
X=data.drop(['status'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)
ypred_kn=model.predict(X_test)
from sklearn.metrics import confusion_matrix
con_kn=confusion_matrix(y_test, ypred_kn)
score = model.score(X_test, y_test)
X_train = X_train.fillna(X_train.median())
X_train.isna().sum()
y_train.isna().sum()

model_random = RandomForestClassifier(n_estimators = 100)
model_random.fit(X_train, y_train)
score_random = model_random.score(X_test, y_test)
ypred_random = model_random.predict(X_test)
con_random = confusion_matrix(y_test, ypred_random)

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state = 0)
logit.fit(X_train, y_train)
ypred_logit = logit.predict(X_test)
score_logit = logit.score(X_test,y_test)
con_logit = confusion_matrix(y_test, ypred_logit)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(X_train, y_train)
y_predsvm = svc.predict(X_test)
score_svm = svc.score(X_test,y_test)
con_svm = confusion_matrix(y_test, y_predsvm)

