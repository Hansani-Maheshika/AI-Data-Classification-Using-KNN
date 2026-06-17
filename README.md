*AI Data Classification Using KNN*

A supervised machine learning web application that classifies Iris flower species using the K-Nearest Neighbors (KNN) algorithm. This project demonstrates the complete beginner-friendly AI classification pipeline, including dataset loading, data preprocessing, train-test splitting, feature scaling, model training, prediction, evaluation, and visualization through a Flask web interface.

Project Overview

This project is developed as part of Artificial Intelligence Project 2: Data Classification Using AI. The main goal is to build a basic classification model using a small dataset and apply a simple supervised learning algorithm.

The project uses the famous Iris dataset, which contains flower measurements such as sepal length, sepal width, petal length, and petal width. Based on these measurements, the trained machine learning model predicts the flower species as one of the following:

Setosa
Versicolor
Virginica

The trained model is integrated into a Flask web application where users can enter flower measurements and receive a prediction result. The application also displays the model evaluation report, confusion matrix, and KNN visualization.

Features
Loads and processes the Iris dataset
Splits data into training and testing sets
Applies feature scaling using StandardScaler
Trains a K-Nearest Neighbors classification model
Predicts Iris flower species from user input
Displays model accuracy and classification report
Visualizes the confusion matrix
Shows a 2D KNN decision boundary visualization
Provides a simple Flask-based web interface
Saves trained model and scaler for reuse
Technologies Used
Python
Flask
Scikit-learn
NumPy
Matplotlib
Joblib
HTML
CSS
Machine Learning Workflow

The project follows a standard supervised machine learning workflow:

Load Dataset
     ↓
Understand Features and Labels
     ↓
Split Data into Training and Testing Sets
     ↓
Apply Feature Scaling
     ↓
Train KNN Model
     ↓
Make Predictions
     ↓
Evaluate Model Performance
     ↓
Visualize Results
     ↓
Deploy with Flask Web App
Dataset

This project uses the built-in Iris dataset from Scikit-learn.

The dataset contains:

150 flower samples
4 input features
3 flower classes
Input Features
Sepal Length
Sepal Width
Petal Length
Petal Width
Output Classes
Setosa
Versicolor
Virginica
Algorithm Used
K-Nearest Neighbors

The project uses the K-Nearest Neighbors (KNN) algorithm.

KNN classifies a new data point by checking the nearest existing data points in the training dataset. In this project, the model uses:

KNeighborsClassifier(n_neighbors=5)

This means the model checks the 5 nearest flower samples and predicts the class based on the majority class among those neighbors.

Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Example model performance:

Accuracy: 0.93

The classification report shows the performance of the model for each flower class.

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.83      1.00      0.91        10
   virginica       1.00      0.80      0.89        10

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
Visualizations
Confusion Matrix

The confusion matrix shows how many predictions were correct and incorrect for each flower class.

It helps identify where the model made mistakes.

Example interpretation:

10 Setosa flowers were correctly classified.
10 Versicolor flowers were correctly classified.
8 Virginica flowers were correctly classified.
2 Virginica flowers were incorrectly classified as Versicolor.
KNN Visualization

The KNN visualization shows how the algorithm separates flower classes using two selected features:

Petal Length
Petal Width

The actual prediction model uses all 4 Iris features, but the 2D visualization is used to clearly explain how KNN separates classes visually.

Project Structure
AI-Data-Classification-Using-KNN/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── knn_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── model_report.txt
│   ├── confusion_matrix.png
│   └── knn_visualization.png
│
├── static/
│   ├── style.css
│   └── plots/
│       ├── confusion_matrix.png
│       └── knn_visualization.png
│
└── templates/
    └── index.html
