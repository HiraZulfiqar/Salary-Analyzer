import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# ==================================================
# STEP 1 — LOAD AND UNDERSTAND THE DATA
# ==================================================

# Load the dataset
df = pd.read_csv("datasets/employees.csv")

print("________ Employee Data ________")
print(df)


# 1. Shape
print("\n________ Shape ________")
print(df.shape)


# 2. Information
print("\n________ Information ________")
print(df.info())


# 3. Statistical Summary
print("\n________ Statistical Summary ________")
print(df.describe())


# 4. Missing Values
print("\n________ Missing Values ________")
print(df.isnull().sum())


# 5. Duplicate Rows
print("\n________ Duplicate Rows ________")
print(df.duplicated().sum())


# ==================================================
# STEP 2 — EDA
# ==================================================

# 1. Gender Count Plot
print("\n________ Gender Count Plot ________")

sns.countplot(data=df, x="Gender")
plt.title("Number of Male and Female Employees")
plt.xlabel("Gender")
plt.ylabel("Number of Employees")
plt.show()


# 2. Department Count Plot
print("\n________ Department Count Plot ________")

sns.countplot(data=df, x="Department")
plt.title("Number of Employees in Each Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.show()


# 3. Salary Histogram
print("\n________ Salary Histogram ________")

sns.histplot(data=df, x="Salary", bins=5)
plt.title("Distribution of Salaries")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.show()


# 4. Salary Box Plot
print("\n________ Salary Box Plot ________")

sns.boxplot(data=df, x="Salary")
plt.title("Salary Distribution and Outliers")
plt.xlabel("Salary")
plt.show()


# 5. Experience vs Salary
print("\n________ Experience vs Salary ________")

sns.scatterplot(data=df, x="Experience", y="Salary")
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# 6. Correlation Heatmap
print("\n________ Correlation Heatmap ________")

correlation = df[["Age", "Experience", "Salary"]].corr()

sns.heatmap(correlation, annot=True)
plt.title("Correlation Between Numerical Features")
plt.show()


# ==================================================
# STEP 3 — PREPARE THE DATA
# ==================================================

# Remove Name
# Name is not useful for predicting salary
df = df.drop("Name", axis=1)


# Separate features (X) and target (y)

X = df.drop("Salary", axis=1)

y = df["Salary"]


print("\n________ X (Features) ________")
print(X)

print("\n________ y (Target) ________")
print(y)


# ==================================================
# STEP 3.1 — TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n________ Training X ________")
print(X_train)

print("\n________ Testing X ________")
print(X_test)

print("\n________ Training y ________")
print(y_train)

print("\n________ Testing y ________")
print(y_test)


# STEP 3.2 — ONE-HOT ENCODING
# ==================================================

categorical_columns = ["Gender", "Department"]

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("encoder", encoder, categorical_columns)
    ],
    remainder="passthrough"
)

# Learn categories from training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Apply the SAME learned categories to test data
X_test_transformed = preprocessor.transform(X_test)

print("\n________ Transformed Training Data ________")
print(X_train_transformed)

print("\n________ Transformed Testing Data ________")
print(X_test_transformed)

print("\n________ Feature Names ________")
print(preprocessor.get_feature_names_out())
# ==================================================
# STEP 3.3 — FEATURE NAMES
# ==================================================

print("\n________ Feature Names ________")

print(preprocessor.get_feature_names_out())

# ==================================================
# STEP 4 — TRAIN THE ML MODEL
# ==================================================

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train_transformed, y_train)

# Make predictions
y_pred = model.predict(X_test_transformed)


# Compare actual vs predicted
print("\n________ Actual Salaries ________")
print(y_test.values)

print("\n________ Predicted Salaries ________")
print(y_pred)

