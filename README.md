# Heart Disease Prediction Using Decision Tree Classifier

## Author
**Edison Garcia**  
Panther ID: 3914748  

## Overview
Heart disease is one of the leading causes of death worldwide. Early prediction can help enable timely medical intervention and improve patient outcomes. This project implements a **Decision Tree Classifier** using the **UCI Heart Disease dataset** to predict the presence of heart disease based on clinical features.

The model was built in Python using the **scikit-learn** library and includes:
- Data preprocessing (scaling)
- Model training and evaluation
- Cross-validation
- Decision tree visualization

## Dataset
The dataset used in this project is a cleaned version of the Cleveland subset of the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

**Features (13 total):**
- age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol), fbs (fasting blood sugar), restecg (resting ECG), thalach (max heart rate), exang (exercise-induced angina), oldpeak, slope, ca, thal

**Target:**
- `0` = No heart disease  
- `1` = Heart disease

## Installation & Requirements

### Python Version
Tested with **Python 3.8+**

### Required Libraries
Install all required packages using:

```bash
pip install -r requirements.txt
How to Run
Ensure heart.csv is in the same directory as the script.

Run the script:

bash
Copy
Edit
python heart_disease_predictor.py
The program will:

Load and preprocess the dataset

Train a Decision Tree model

Output evaluation metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

5-Fold Cross-Validation Accuracy

Visualize the decision tree structure in a pop-up window

Sample Output
yaml
Copy
Edit
✅ Dataset loaded successfully!
Shape of dataset: (303, 14)
Accuracy: 0.8196
Precision: 0.8181
Recall: 0.8437
F1 Score: 0.8307
Confusion Matrix:
 [[23  6]
  [ 5 27]]
5-Fold Cross-Validation Accuracy: 0.7821
Project Structure
Copy
Edit
.
├── heart.csv
├── heart_disease_predictor.py
├── requirements.txt
└── README.md
Acknowledgements
UCI Machine Learning Repository

scikit-learn documentation

yaml
Copy
Edit

---


