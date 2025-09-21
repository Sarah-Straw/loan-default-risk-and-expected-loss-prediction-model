# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 17:46:58 2025

Loan Default Probability and Expected Loss Calculator

@author: Sarah

"""
#==============================================================================
# Imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

#==============================================================================
# Funtions

# expected loss equation
def expected_loss(PD, recovery_rate, exposure):
    
    return PD * (1-recovery_rate) * exposure

# trained model gives probability of default 
def PD_calculator(model, credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score):
    array = np.array([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]])
    PD = model.predict_proba(array)[:, 1][0] 
    return PD 
    
#==============================================================================
# Importing and organising data 

df = pd.read_excel("Task_3_and_4_Loan_Data.xlsx") 

variables = ["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding", "income", "years_employed", "fico_score"]

X = df[variables]

Y = df["default"]

#==============================================================================
# Analysis

# randomly splitting data in 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)

# regression model trained with training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]   # probability of default

auc = roc_auc_score(Y_test, y_pred_prob) # area under the ROC curve

#==============================================================================
# Output

print("")
print("AUC:", auc) 
print("")
print(classification_report(Y_test, model.predict(X_test)))
print("")

# input values IN ORDER for: credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score
# I've used a line of the data as an example

pd = PD_calculator(model, X.iloc[1,0], X.iloc[1,1], X.iloc[1,2], X.iloc[1,3], X.iloc[1,4], X.iloc[1,5]) 

exposure = example_exposure = 10000

exp_loss = expected_loss(pd, 0.1, exposure)
print("Expected Loss: £", exp_loss )












