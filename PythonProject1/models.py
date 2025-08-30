# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
df = pd.read_csv(r"C:\Users\HEDY HEDAR\PycharmProjects\PythonProject1\data\creditcard.csv")

# Step 3: Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)

# Step 5: Scale "Time" and "Amount"
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

# Step 6A: Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)
y_pred_log  = log_model.predict(X_test)
y_prob_log  = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Step 6B: Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

print("\nDecision Tree Results:")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Step 7: Confusion Matrix Heatmaps
# Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6,4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Decision Tree
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(6,4))
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Step 8: ROC Curve Comparison
fpr_log,  tpr_log,  _ = roc_curve(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_log  = auc(fpr_log,  tpr_log)
auc_tree = auc(fpr_tree, tpr_tree)

plt.figure(figsize=(6,4))
plt.plot(fpr_log,  tpr_log,  label=f'LogReg (AUC = {auc_log:.2f})')
plt.plot(fpr_tree, tpr_tree, label=f'DecTree (AUC = {auc_tree:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
