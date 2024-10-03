# Multi-Output Neural Network for Attrition and Department Prediction

# Project Overview

This project builds a multi-output neural network model to predict two outcomes simultaneously:

Attrition: Whether an employee will leave the company or not.
Department: The department to which an employee belongs.
The model is trained on a dataset containing employee features and uses these features to predict both attrition (a classification task) and department (another classification task). The project focuses on improving the predictive power of the model by experimenting with different model architectures, feature engineering, and optimization techniques.

# Project Goals

* **Develop a multi-output model** that can predict two labels (attrition and department) at the same time.
* **Optimize prediction accuracy** for both tasks through techniques like feature engineering, regularization, and model tuning.
* Handle imbalanced data (if applicable) to ensure that predictions, especially for attrition, are meaningful and reliable.
# Dataset
The dataset contains various employee-related features (e.g., job role, salary, performance ratings, etc.) used to predict two target variables:
* **Attrition:** Whether an employee stays or leaves the company (binary classification).
* **Department:** The department to which the employee belongs (multi-class classification).

# Data Structure
* **Features:** Employee demographics, job details, and other attributes (e.g., age, salary, job satisfaction).
* **Targets:**
  * **Attrition:** Binary target (0 for staying, 1 for leaving).
  * **Department:** Multi-class target representing different departments within the organization.

# Model Architecture

The model is built using Keras with a multi-output neural network design. The structure includes:

* **Input Layer:** Accepts employee features as inputs.
* **Hidden Layers:** Several fully connected layers with ReLU activation, batch normalization, and dropout to prevent overfitting.
* **Output Layers:**
  * **Department Output:** A softmax layer for multi-class classification of departments.
  * **Attrition Output:** A softmax layer for binary classification of employee attrition.

# Compilation

The model uses the Adam optimizer and the categorical cross-entropy loss for both outputs, along with accuracy as the performance metric.

model = Model(inputs=input_layer, outputs=[dept_output_layer, attrition_output_layer])
# Compile the model
model.compile(
    optimizer="adam",
    loss={
        "department_output": "categorical_crossentropy",
        "attrition_output": "categorical_crossentropy"  # Changed from 'binary_crossentropy'
    },
    metrics={
        "department_output": "accuracy",
        "attrition_output": "accuracy"
    }
)

)
# Model Training
The model is trained on the employee data, using features as inputs and predicting the two targets—attrition and department—simultaneously. The training process involves:

Training Data: Split into training and testing sets.
Batch Size: 32
Epochs: 20 (with early stopping and model validation)

history = model.fit(
    X_train,
    {
        "department_output": y_train_dept_encoded,
        "attrition_output": y_train_attrition_encoded
    },
    epochs=20,
    batch_size=32,
    validation_data=(
        X_test,
        {
            "department_output": y_test_dept_encoded,
            "attrition_output": y_test_attrition_encoded
        }
    )
)

**Results**

* Department Prediction Accuracy: 52.7%
* Attrition Prediction Accuracy: 82.6%

# How to Run

1. Clone the repository and navigate to the project directory.
2. Install required dependencies (Python 3.7+ recommended):
3. Train the model using the dataset (replace X_train and y_train with your actual data).
4. Evaluate the model on test data:

# Future Work
Improve Department Accuracy: Experiment with additional features and advanced neural network architectures to boost department prediction accuracy.
Model Tuning: Further fine-tuning of hyperparameters to improve performance.
Class Imbalance: Explore techniques like class weighting and oversampling for handling imbalanced data more effectively.

# Dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
