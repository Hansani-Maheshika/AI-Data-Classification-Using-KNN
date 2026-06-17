# AI Data Classification Using KNN

A supervised machine learning project that classifies Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm. The project includes model training, evaluation, visualization, and a simple Flask web application for prediction.

## Project Overview

This project uses the Iris dataset to classify flowers into three categories:

* Setosa
* Versicolor
* Virginica

The model predicts the flower species using four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## Technologies Used

* Python
* Flask
* Scikit-learn
* NumPy
* Matplotlib
* Joblib
* HTML
* CSS

## Machine Learning Workflow

```text
Load Iris Dataset
Split Data into Training and Testing Sets
Apply Feature Scaling
Train KNN Model
Evaluate Model Performance
Show Results in Flask App
```

## Features

* Iris flower classification using KNN
* Train-test split
* Feature scaling with StandardScaler
* Model evaluation report
* Confusion matrix visualization
* KNN 2D visualization
* Flask web interface for prediction

## Project Structure

```text
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
```



## Conclusion

This project demonstrates a complete basic AI classification workflow using the KNN algorithm and Flask. It helps understand supervised learning, model training, prediction, and evaluation.
