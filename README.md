# Task2
Import libraries and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('loan_prediction.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing data appropriately (e.g., fill with mean, median, or mode)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Drop rows with missing values in other columns if any
df.dropna(inplace=True)
# Visualize loan amount distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['LoanAmount'], bins=20, kde=True)
plt.title('Loan Amount Distribution')
plt.show()

# Visualize education
plt.figure(figsize=(6, 4))
sns.countplot(x='Education', data=df)
plt.title('Education Level')
plt.show()

# Visualize income
plt.figure(figsize=(8, 6))
sns.histplot(df['ApplicantIncome'], bins=20, kde=True)
plt.title('Applicant Income Distribution')
plt.show()
# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Education'] = le.fit_transform(df['Education'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Property_Area'] = le.fit_transform(df['Property_Area'])

# Define features and target
X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Train a decision tree classifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
