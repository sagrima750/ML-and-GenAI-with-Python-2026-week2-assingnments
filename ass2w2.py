# ==========================
# Netflix User Analytics
# ==========================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------
# Q1: Load Dataset
# --------------------------
df = pd.read_csv("netflix_users.csv")

print("First Five Records:")
print(df.head())

# --------------------------
# Q2: Number of Rows & Columns
# --------------------------
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# --------------------------
# Q3: Display Column Names
# --------------------------
print("\nColumn Names:")
print(df.columns)

# --------------------------
# Q4: Numerical & Categorical Features
# --------------------------
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

print("\nNumerical Features:")
print(list(numerical_features))

print("\nCategorical Features:")
print(list(categorical_features))

# --------------------------
# Q5: Missing Values
# --------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ====================================================
# Part B: Exploratory Data Analysis
# ====================================================

# Q6: Average Age
print("\nAverage Age:", df['Age'].mean())

# Q7: Average Watch Hours Per Week
print("Average Watch Hours Per Week:", df['WatchHoursPerWeek'].mean())

# Q8: Average Monthly Spending
print("Average Monthly Spending:", df['MonthlySpend'].mean())

# Q9: Users in Each Subscription Category
print("\nSubscription Type Counts:")
print(df['SubscriptionType'].value_counts())

# Q10: Percentage of Renewed Subscriptions
renewed_percentage = (
    (df['SubscriptionRenewed'] == 'Yes').sum()
    / len(df)
) * 100

print("\nRenewal Percentage:", round(renewed_percentage, 2), "%")

# ====================================================
# Part C: Data Preparation
# ====================================================

# Q11: Encode Categorical Features
df_encoded = df.copy()

label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

print("\nEncoded Dataset:")
print(df_encoded.head())

# Q12: Define Features (X) and Target (y)
X = df_encoded.drop(['UserID', 'SubscriptionRenewed'], axis=1)
y = df_encoded['SubscriptionRenewed']

# Q13: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ====================================================
# Part D: Decision Tree Classification
# ====================================================

# Q14: Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Q15: Accuracy
dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")

# Q16: Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_predictions)

print("\nDecision Tree Confusion Matrix:")
print(dt_cm)

# ====================================================
# Part E: K-Nearest Neighbors (KNN)
# ====================================================

# Q17: Train KNN (K=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

# Q18: Compare Accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nKNN Accuracy:", round(knn_accuracy * 100, 2), "%")

print("\nAccuracy Comparison")
print("-------------------")
print("Decision Tree:", round(dt_accuracy * 100, 2), "%")
print("KNN          :", round(knn_accuracy * 100, 2), "%")

if dt_accuracy > knn_accuracy:
    print("Decision Tree performs better.")
elif knn_accuracy > dt_accuracy:
    print("KNN performs better.")
else:
    print("Both models perform equally.")

# ====================================================
# Part F: Linear Regression
# ====================================================

# Q19: Train Linear Regression to Predict MonthlySpend
X_reg = df_encoded.drop(['UserID', 'MonthlySpend'], axis=1)
y_reg = df_encoded['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

# Q20: Predict Monthly Spending for a New User

new_user = pd.DataFrame({
    'Age': [30],
    'Gender': [1],               # Example encoded value
    'SubscriptionType': [2],     # Example encoded value
    'WatchHoursPerWeek': [20],
    'DevicesUsed': [3],
    'FavoriteGenre': [1],        # Example encoded value
    'AdClicks': [10],
    'SubscriptionRenewed': [1]
})

predicted_spend = lr_model.predict(new_user)

print("\nPredicted Monthly Spending for New User:")
print("₹", round(predicted_spend[0], 2))

print("\nInterpretation:")
print(
    f"The model predicts that this user is likely to spend "
    f"approximately ₹{round(predicted_spend[0],2)} per month on Netflix."
)