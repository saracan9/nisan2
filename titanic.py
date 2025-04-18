import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train.csv")

print("Veri şekli:", train_df.shape)
print(train_df.head())
print("\nVeri tipi ve eksik değer kontrolü:")
train_df.info()
print("\nEksik veri sayısı:")
print(train_df.isnull().sum())

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())  
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])  
train_df.drop("Cabin", axis=1, inplace=True)  

train_df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)


le = LabelEncoder()
train_df["Sex"] = le.fit_transform(train_df["Sex"])        
train_df["Embarked"] = le.fit_transform(train_df["Embarked"])  
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(" Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\n Karışıklık Matrisi (Confusion Matrix):\n", confusion_matrix(y_test, y_pred))
print("\n Sınıflandırma Raporu (Classification Report):\n", classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  
}


grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)


best_model = grid.best_estimator_
print(" En iyi parametreler:", grid.best_params_)


y_pred_optimized = best_model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n Yeni Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("\n Yeni Confusion Matrix:\n", confusion_matrix(y_test, y_pred_optimized))
print("\n Yeni Classification Report:\n", classification_report(y_test, y_pred_optimized))


from sklearn.neighbors import KNeighborsClassifier

knn_param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_param_grid, cv=5, scoring='f1')
knn_grid.fit(X_train, y_train)

best_knn = knn_grid.best_estimator_
print("\n KNN için en iyi parametreler:", knn_grid.best_params_)

y_pred_knn = best_knn.predict(X_test)

print("\n KNN Doğruluk (Accuracy):", accuracy_score(y_test, y_pred_knn))
print("\n KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\n KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
