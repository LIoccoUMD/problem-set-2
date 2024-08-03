'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

# Your code here
def load_data():
    pred_universe_df = pd.read_csv("src/data/pred_universe_raw.csv")
    arrest_events_df = pd.read_csv("src/data/arrest_events_raw.csv")
    return pred_universe_df, arrest_events_df

def merge_data(df1, df2):
    df_arrests = pd.merge(df1, df2, on="person_id", how="outer")
    return df_arrests

def add_y_column(df):
    # Ensure arrest dates are datetime objects
    df['arrest_date_univ'] = pd.to_datetime(df['arrest_date_univ'])
    df['arrest_date_event'] = pd.to_datetime(df['arrest_date_event'])

    # Create a new column 'felony_within_365_days'
    df['felony_within_365_days'] = df.apply(
        lambda row: any(
            (df['person_id'] == row['person_id']) &
            (df['arrest_date_event'] > row['arrest_date_event']) &
            (df['arrest_date_event'] <= row['arrest_date_event'] + pd.Timedelta(days=365)) &
            (df['charge_degree'] == 'felony')
        ),
        axis=1
    )

    # Map 'felony_within_365_days' to 'y'
    df['y'] = df['felony_within_365_days'].map({True: 1, False: 0})
    return df

def share_of_arrestees(df):
    # Calculate the share of arrestees rearrested for a felony within the following year
    share_rearrested_for_felony = df['y'].mean()
    print(f"\n{share_rearrested_for_felony:.2%} of arrestees were rearrested for a felony crime in the following year.")

# def main():
#     pred_universe_df, arrest_events_df = load_data()
#     df_arrests = merge_data(pred_universe_df,arrest_events_df)
#     merge_data(pred_universe_df, arrest_events_df)
#     add_y_column(df_arrests)
    
#     print("Pred Universe DataFrame:")
#     print(pred_universe_df.head())
        
#     print("\nArrest Events DataFrame:")
#     print(arrest_events_df.head())
    
#     print("\nMerged DataFrame (df_arrests):")
#     print(df_arrests.head(500))
    
#     print("\nMerged DataFrame (df_arrests) with 'y' column:")
#     print(df_arrests.head())
    
#     share_of_arrestees(df_arrests)

    
# if __name__ == "__main__":
#     main()