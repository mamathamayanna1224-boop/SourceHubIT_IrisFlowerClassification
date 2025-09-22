# Iris Flower Classification with IRIS.csv

# 1. Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset
data = pd.read_csv("IRIS.csv")

print("First 5 rows of dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# 3. EDA (Exploratory Data Analysis)
sns.pairplot(data, hue="species")
plt.show()

# 4. Prepare features and labels
X = data.drop("species", axis=1)   # Features (Sepal length, Sepal width, etc.)
y = data["species"]                # Target (setosa, versicolor, virginica)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))