
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Basic info
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

# Fraud vs Non-fraud Distribution
print(df["Class"].value_counts())
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Count")
plt.show()

# Percentage version
counts = df["Class"].value_counts(normalize=True) * 100
ax = sns.barplot(x=counts.index, y=counts.values)
plt.title("Fraud vs Non-Fraud Percentage")
plt.ylabel("Percentage (%)")

# Add labels on top of bars
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha='center')

plt.show()

# Transaction Amount by Class (Boxplot)
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Fraud/Non-Fraud")
plt.show()

# Distribution of Amounts for Fraud vs Non-Fraud
sns.histplot(df[df["Class"] == 0]["Amount"], bins=100, color="blue", label="Non-Fraud", kde=True)
sns.histplot(df[df["Class"] == 1]["Amount"], bins=100, color="red", label="Fraud", kde=True)
plt.legend()
plt.title("Transaction Amount Distribution")
plt.show()

# Distribution of transactions over time
sns.histplot(df[df["Class"] == 0]["Time"], bins=100, color="blue", label="Non-Fraud")
sns.histplot(df[df["Class"] == 1]["Time"], bins=100, color="red", label="Fraud")
plt.legend()
plt.title("Transactions Over Time")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

print(df.describe())








