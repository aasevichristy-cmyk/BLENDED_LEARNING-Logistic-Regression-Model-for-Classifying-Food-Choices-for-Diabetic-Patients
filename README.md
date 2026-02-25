# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Explore the Dataset
2. Preprocess the Data
3. Split the Dataset
4. Train the Logistic Regression Model
5. Evaluate the Model


## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Anisha A
RegisterNumber:  212225220009
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('food_items (1).csv')
print('Name:Anisha A ')
print('Reg. No: 212225220009')
print('Dataset Overview:')
print(df.head())
print('\nDataset Info:')
print(df.info())
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000
l2_model = LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver, max_iter=max_iter)
l2_model.fit(X_train, y_train)
y_pred = l2_model.predict(X_test)
print('Name:Anisha A ')
print('Reg. No: 212225220009')
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print('Name: Anisha A')
print('Reg. No: 212225220009')

```

## Output:

<img width="1060" height="656" alt="Screenshot 2026-02-25 143006" src="https://github.com/user-attachments/assets/d4ca750c-c060-4b3e-8636-117b3895170e" />

<img width="814" height="595" alt="Screenshot 2026-02-25 143137" src="https://github.com/user-attachments/assets/9539ce5d-4e7b-4e94-b236-010ffa317ec1" />

<img width="1109" height="94" alt="Screenshot 2026-02-25 143203" src="https://github.com/user-attachments/assets/0f886d7e-48d7-4838-a27d-c1b58426ade5" />

<img width="814" height="350" alt="Screenshot 2026-02-25 143232" src="https://github.com/user-attachments/assets/64b07d74-3096-4832-b8a8-986d46648a40" />

<img width="326" height="128" alt="Screenshot 2026-02-25 143300" src="https://github.com/user-attachments/assets/1751938f-b311-47da-818d-3d7724f9d9b5" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
