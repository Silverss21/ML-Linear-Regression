# Breast Cancer Diagnosis Prediction Using Logistic Regression

This repository contains a Python script that demonstrates the process of predicting breast cancer diagnosis using logistic regression. The dataset used for this demonstration is from `breast-cancer.csv`.

## Dataset Information

The dataset includes various attributes related to breast cancer tumors, with the target variable being the diagnosis ('M' for malignant and 'B' for benign).

### Data Cleaning Steps

- The script begins by loading the dataset and handling missing values by replacing zeros with NaN and dropping rows with NaN values.
- It maps the diagnosis column to binary labels: 'M' (malignant) mapped to 1 and 'B' (benign) mapped to -1.

## Steps Taken in the Script

1.  **Loading and Exploring the Dataset**:

    - Reads the CSV file `breast-cancer.csv`.
    - Displays the first 12 rows and the last 5 rows of the dataset.
    - Prints the dimensions of the dataset.

2.  **Handling Missing Values**:

    - Identifies and counts missing values in each column.
    - Checks for missing values across the entire dataset.

3.  **Data Preprocessing**:

    - Drops the 'id' column if present, as it is not a feature for prediction.
    - Encodes the target variable (`diagnosis`) into binary labels.

4.  **Splitting the Data**:

    - Splits the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.

5.  **Model Training and Evaluation**:

    - Initializes a logistic regression model and fits it using the training data.
    - Makes predictions on the test data and evaluates the model's performance using accuracy score, confusion matrix, and classification report.

6.  **Results**:

    - Displays actual vs. predicted values for the first 10 test samples.
    - Prints the confusion matrix.
    - Outputs the accuracy score and classification report summarizing the model's performance.

## Running the Code

To run the code, follow these steps:

1.  Ensure Python and necessary libraries (`pandas`, `numpy`, `scikit-learn`) are installed.
2.  Download the `breast-cancer.csv` dataset and place it in the same directory as the script.
3.  Execute the script `breast_cancer_prediction.py`.

```
python breast_cancer_prediction.py
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Additional Notes

- Increase the `max_iter` parameter in `LogisticRegression()` if convergence warnings occur during model fitting.
- Adjust `test_size` in `train_test_split` to modify the ratio of training and testing data.
