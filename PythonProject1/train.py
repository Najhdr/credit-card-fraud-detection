# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load dataset
# We now use the Kaggle credit card fraud dataset
df = pd.read_csv(r"/Data/creditcard.csv")

# Step 3: Split into features (X) and target (y)
# X = all columns except "Class"
# y = "Class" (0 = not fraud, 1 = fraud)
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Train/Test Split
# Training set = 70%, Testing set = 30%
# "stratify=y" keeps fraud ratio the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)

# Step 5: Scale the "Time" and "Amount" columns
# Scaling helps the model handle different ranges of values
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# Step 6: Train a Logistic Regression model
# Logistic Regression is a simple baseline model
# class_weight="balanced" tells it to care more about fraud (since fraud is rare)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
# Confusion Matrix shows how many frauds were caught or missed
# Classification Report shows precision, recall, and F1-score
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
