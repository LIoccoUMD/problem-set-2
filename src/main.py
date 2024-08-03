'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
# import logistic_regression
# import decision_tree
# import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw, arrest_events_raw = part1_etl.etl_process()
    # PART 2: Call functions/instanciate objects from preprocessing
    part2_preprocessing.preprocess_data()
    df = part2_preprocessing
    # PART 3: Call functions/instanciate objects from logistic_regression

    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()