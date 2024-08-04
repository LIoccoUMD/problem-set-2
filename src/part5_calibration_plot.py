'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import part3_logistic_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=5):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set_theme(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()
    
    
def compare_calibration(df_arrests_test, gs_cv_lr, gs_cv_dt, features):
    # Get true labels and predicted probabilities for the test set
    y_true = df_arrests_test['y']
    x_test = df_arrests_test[features]
    
    # Calibrate the Logistic Regression model
    calibrated_lr = CalibratedClassifierCV(gs_cv_lr, cv='prefit')
    calibrated_lr.fit(x_test, y_true)
    y_prob_lr = calibrated_lr.predict_proba(x_test)[:, 1]
    
    # Calibrate the Decision Tree model
    calibrated_dt = CalibratedClassifierCV(gs_cv_dt, cv='prefit')
    calibrated_dt.fit(x_test, y_true)
    y_prob_dt = calibrated_dt.predict_proba(x_test)[:, 1]
    
    # Plot calibration curves
    calibration_plot(y_true, y_prob_lr, 'Logistic Regression')
    calibration_plot(y_true, y_prob_dt, 'Decision Tree')
    
    # Print which model is more calibrated (visually inspect the plots)
    print("Which model is more calibrated? This can be evaluated visually from the calibration plots.")