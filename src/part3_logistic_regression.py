'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import sys
import os
import pandas as pd
import numpy as np
from part2_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Read in df from part2 directly
def load_data():
    with SuppressPrint():
        df_arrests = preprocess_data()
    print(df_arrests)
    return df_arrests

def split_data(df):
    df_arrests_train, df_arrests_test = train_test_split(df, test_size=0.3, shuffle=True, stratify=df['y'])
    return df_arrests_train, df_arrests_test

def prepare_and_run_model():
    features = ["current_charge_felony", "num_fel_arrests_last_year"]
    param_grid = {"C": [0.1,1,10]}
    lr_model = lr()
    gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5)
    return features, param_grid, lr_model, gs_cv

    #Change main to logistic_regression when finished.
def main():
    df_arrests = load_data()
    print("Columns in df_arrests: ", df_arrests.columns.tolist())
    df_arrests_train, df_arrests_test = split_data(df_arrests)
    features, param_grid, lr_model, gs_cv = prepare_and_run_model()
    
    x_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    gs_cv.fit(x_train,y_train)
    
    print("Training DataFrame:")
    print(df_arrests_train)
    print("\nTesting DataFrame:")
    print(df_arrests_test)
    
    print("Features:", features)
    print("Parameter grid:", param_grid)
    print("Logistic Regression model initialized.")
    print("GridSearchCV initialized with 5-fold cross-validation.")
    print("GridSearchCV run completed.")
    return df_arrests_train, df_arrests_test, features, param_grid, lr_model, gs_cv

if __name__ == "__main__":
    main()