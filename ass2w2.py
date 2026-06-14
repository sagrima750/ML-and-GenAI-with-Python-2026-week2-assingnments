# ==========================================
# Netflix User Analytics Assignment
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# Q1: Load Dataset
# ==========================================

df = pd.read_csv("netflix_users.csv")

print("First Five Records:")
print(df.head())

# ==========================================
# Q2: Rows and Columns
# ==========================================

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# ==========================================
# Q3: Column Names
# ==========================================

print("\nColumn Names:")
print(df.columns)

# ==========================================
# Q4: Numerical and Categorical Features
# ==========================================

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object', 'string']).columns

print("\nNumerical Features:")
print(list(numerical_features))

print("\nCategorical Features:")
print(list(categorical_features))

# ==========================================
# Q5: Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Part B: Exploratory Data Analysis
# ==========================================

# Q6
print("\nAverage Age:", round(df['Age'].mean(), 2))

# Q7
print("Average Watch Hours Per Week:",
      round(df['WatchHoursPerWeek'].mean(), 2))

# Q8
print("Average Monthly Spending:",
      round(df['MonthlySpend'].mean(), 2))

# Q9
print("\nSubscription Type Counts:")
print(df['SubscriptionType'].value_counts())

# Q10
renewed_percentage = (
    df['SubscriptionRenewed']
    .astype(str)
    .str.lower()
    .eq('yes')
    .mean()
) * 100

print("\nRenewal Percentage:",
      round(renewed_percentage, 2), "%")

# ==========================================
# Part C: Data Preparation
# ==========================================

# Q11 Encode Categorical Features

df_encoded = df.copy()

label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

print("\nEncoded Dataset:")
print(df_encoded.head())

# Q12 Define X and y

X = df_encoded.drop(['UserID', 'SubscriptionRenewed'], axis=1)
y = df_encoded['SubscriptionRenewed']

# Q13 Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# Part D: Decision Tree
# ==========================================

# Q14 Train Decision Tree

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Q15 Accuracy

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

# Q16 Confusion Matrix

dt_cm = confusion_matrix(
    y_test,
    dt_predictions,
    labels=[0, 1]
)

print("\nDecision Tree Confusion Matrix:")
print(dt_cm)

# ==========================================
# Part E: KNN
# ==========================================

# Q17 Train KNN

k = min(5, len(X_train))

knn_model = KNeighborsClassifier(n_neighbors=k)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

# Q18 Accuracy Comparison

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nKNN Accuracy:",
      round(knn_accuracy * 100, 2), "%")

print("\nAccuracy Comparison")
print("-------------------")
print("Decision Tree:",
      round(dt_accuracy * 100, 2), "%")

print("KNN:",
      round(knn_accuracy * 100, 2), "%")

if dt_accuracy > knn_accuracy:
    print("Decision Tree performs better.")
elif knn_accuracy > dt_accuracy:
    print("KNN performs better.")
else:
    print("Both models perform equally.")

# ==========================================
# Part F: Linear Regression
# ==========================================

# Q19 Train Linear Regression

X_reg = df_encoded.drop(
    ['UserID', 'MonthlySpend'],
    axis=1
)

y_reg = df_encoded['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

# ==========================================
# Q20 Predict Monthly Spending
# ==========================================

new_user = pd.DataFrame({
    'Age': [30],
    'Gender': [1],
    'SubscriptionType': [2],
    'WatchHoursPerWeek': [20],
    'DevicesUsed': [3],
    'FavoriteGenre': [1],
    'AdClicks': [10],
    'SubscriptionRenewed': [1]
})

predicted_spend = lr_model.predict(new_user)

print("\nPredicted Monthly Spending for New User:")
print("Rs.", round(predicted_spend[0], 2))

print("\nInterpretation:")
print(
    f"The model predicts that this user is likely to spend "
    f"approximately Rs. {round(predicted_spend[0], 2)} "
    f"per month on Netflix."
)

print("\nProgram Executed Successfully.")
