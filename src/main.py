'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw, arrest_events_raw = part1_etl.etl_process()
    # PART 2: Call functions/instanciate objects from preprocessing
    part2_preprocessing.preprocess_data()
    # PART 3: Call functions/instanciate objects from logistic_regression
    part3_logistic_regression.logistic_regression()
    # PART 4: Call functions/instanciate objects from decision_tree
    part4_decision_tree.decision_tree()
    # PART 5: Call functions/instanciate objects from calibration_plot
    df_arrests_train, df_arrests_test, features, param_grid, lr_model, gs_cv_lr, best_c = part3_logistic_regression.logistic_regression()
    df_arrests_test, gs_cv_dt = part4_decision_tree.decision_tree_model(df_arrests_train, df_arrests_test, features)
    part5_calibration_plot.CalibratedClassifierCV(df_arrests_test, gs_cv_lr, gs_cv_dt, features)

if __name__ == "__main__":
    main()