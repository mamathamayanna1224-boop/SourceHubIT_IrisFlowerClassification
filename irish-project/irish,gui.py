import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("IRIS.csv")

X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1 = report['weighted avg']['f1-score']

# GUI
def predict_species():
    try:
        sl = float(entry_sl.get())
        sw = float(entry_sw.get())
        pl = float(entry_pl.get())
        pw = float(entry_pw.get())

        prediction = model.predict([[sl, sw, pl, pw]])[0]

        result_label.config(text=f"🌸 Predicted Species: {prediction}")
        metrics_label.config(
            text=f"📊 Model Performance:\n"
                 f"Accuracy: {accuracy:.2f}\n"
                 f"Precision: {precision:.2f}\n"
                 f"Recall: {recall:.2f}\n"
                 f"F1-Score: {f1:.2f}"
        )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

# Create window
root = tk.Tk()
root.title("Iris Species Predictor")
root.geometry("400x450")

tk.Label(root, text="Sepal Length (cm):").pack()
entry_sl = tk.Entry(root)
entry_sl.pack()

tk.Label(root, text="Sepal Width (cm):").pack()
entry_sw = tk.Entry(root)
entry_sw.pack()

tk.Label(root, text="Petal Length (cm):").pack()
entry_pl = tk.Entry(root)
entry_pl.pack()

tk.Label(root, text="Petal Width (cm):").pack()
entry_pw = tk.Entry(root)
entry_pw.pack()

tk.Button(root, text="Predict Species", command=predict_species, bg="green", fg="white").pack(pady=10)

# Initialize the labels *before* using .config()
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="blue")
result_label.pack(pady=5)

metrics_label = tk.Label(root, text="", font=("Arial", 10), justify="left")
metrics_label.pack(pady=10)

root.mainloop()
