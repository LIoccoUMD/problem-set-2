'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
import part3_logistic_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def decision_tree_model(df_train, df_test, features):
    param_grid_dt = {'max_depth':[3,5,7]}
    dt_model = DTC()
    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5)
    x_train = df_train[features]
    y_train = df_train['y']
    gs_cv_dt.fit(x_train,y_train)
    
    # Predictions
    x_test = df_test[features]
    pred_dt = gs_cv_dt.predict(x_test)
    df_test["pred_dt"] = pred_dt
    return df_test, gs_cv_dt

def decision_tree():
    
    df_arrests_train, df_arrests_test, features, param_grid, lr_model, gs_cv, best_c = part3_logistic_regression.logistic_regression()
    

    df_arrests_test, gs_cv_dt = decision_tree_model(df_arrests_train, df_arrests_test, features)
    print(df_arrests_test)
    

    best_params_dt = gs_cv_dt.best_params_
    best_depth = best_params_dt['max_depth']
    print(f"\nWhat was the optimal tree depth?\n\tThe optimal tree depth is {best_depth}.")
    
    return df_arrests_test, gs_cv_dt