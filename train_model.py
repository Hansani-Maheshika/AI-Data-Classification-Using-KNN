import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature Names:", feature_names)
print("Target Names:", target_names)
print("Dataset Shape:", X.shape)

# Split full dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale full dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train main KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Save model and scaler
joblib.dump(model, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Save text report
with open("outputs/model_report.txt", "w") as file:
    file.write("AI Data Classification Using KNN\n")
    file.write("================================\n\n")
    file.write(f"Accuracy: {accuracy:.2f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)

# Save confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
disp.plot(ax=ax)
plt.title("Confusion Matrix - KNN Iris Classification")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.savefig("static/plots/confusion_matrix.png")
plt.close(fig)

# -----------------------------
# KNN VISUALIZATION (2D only)
# -----------------------------
# Use only 2 features for visualization:
# petal length and petal width
X_vis = iris.data[:, [2, 3]]
y_vis = iris.target

scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_vis_scaled, y_vis)

# Create mesh grid
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

scatter = plt.scatter(
    X_vis_scaled[:, 0],
    X_vis_scaled[:, 1],
    c=y_vis,
    edgecolor="k"
)

plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("KNN Visualization (2D)")

handles, _ = scatter.legend_elements()
plt.legend(handles, target_names, title="Classes")
plt.tight_layout()

plt.savefig("outputs/knn_visualization.png")
plt.savefig("static/plots/knn_visualization.png")
plt.close()

print("Model training completed successfully.")
print("Model saved inside model folder.")
print("Reports and plots saved successfully.")