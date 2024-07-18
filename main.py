import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression



cancer = pd.read_csv('C:\AI-class/breast-cancer.csv')

cancer.head(12)

cancer.tail(5)

num_rows, num_cols = cancer.shape
print(f"The dataset has {num_rows} rows and {num_cols} columns.")

cancer.replace(0, np.nan, inplace=True)
missing_values = cancer.isnull().sum()

# Print the result
print("Missing values in each column:")
print(missing_values)

# Check if there are any missing values in the entire DataFrame
has_missing_values = cancer.isnull().values.any()

if has_missing_values:
    print("\nThe dataset contains missing values.")
else:
    print("\nThe dataset does not contain any missing values.")
    

# Drop the 'id' column if it exists, as it is not a feature
if 'id' in cancer.columns:
    cancer = cancer.drop(columns=['id'])

# Encode the target variable
cancer['diagnosis'] = cancer['diagnosis'].map({'M': 1, 'B': -1})

if cancer['diagnosis'].isnull().any():
    print("There are NaN values in the diagnosis column after mapping. Please check the dataset.")
    print(cancer['diagnosis'].isnull())
else:
    print("Mapping of diagnosis column is successful.")

cancer_cleaned = cancer.dropna()

X = cancer_cleaned.drop(columns=['diagnosis'])
Y = cancer_cleaned['diagnosis']




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=10000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Make predictions
Y_pred = model.predict(X_test)

print("Actual vs. Predicted values for the first 10 test samples:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
    
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)