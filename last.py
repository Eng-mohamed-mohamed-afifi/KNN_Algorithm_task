

# Random Forest Classification
# Decision Tree 
# Logistic Regression
# KNN

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

Random_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
Random_classifier.fit(X_train, y_train)

# Predicting the Test set results
Random_y_pred = Random_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, mean_absolute_error

random_cm = confusion_matrix(y_test, Random_y_pred)
# print(random_cm)
random_ac = (random_cm[0][0] + random_cm[1][1]) / (len(y_test))
print(f"The Random Forest model accuracy {random_ac * 100} %")
random_error = mean_absolute_error(y_test, Random_y_pred)
print(f"The Random Forest model error {random_error * 100} %")
print("-------------------------------------")

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

Logistic_classifier = LogisticRegression()
Logistic_classifier.fit(X_train, y_train)
# Predicting the Test set results
logistic_y_pred = Logistic_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, mean_absolute_error

logistic_cm = confusion_matrix(y_test, logistic_y_pred)
# print(logistic_cm)
logistic_ac = (logistic_cm[0][0] + logistic_cm[1][1]) / (len(y_test))
print(f"The Logistic Regression model accuracy {logistic_ac * 100} %")
logistic_error = mean_absolute_error(y_test, logistic_y_pred)
print(f"The Logistic Regression model error {logistic_error * 100} %")
print("-------------------------------------")

# Fitting Decision Tree  to the Training set
from sklearn.tree import DecisionTreeClassifier

Decision_classifier = DecisionTreeClassifier()
Decision_classifier.fit(X_train, y_train)
# Predicting the Test set results
decision_y_pred = Decision_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, mean_absolute_error

decision_cm = confusion_matrix(y_test, decision_y_pred)
# print(logistic_cm)
decision_ac = (decision_cm[0][0] + decision_cm[1][1]) / (len(y_test))
print(f"The Decision Tree model accuracy {decision_ac * 100} %")
decision_error = mean_absolute_error(y_test, decision_y_pred)
print(f"The Decision Tree model error {decision_error * 100} %")
print("-------------------------------------")

# Fitting KNN  to the Training set
from sklearn.neighbors import KNeighborsClassifier

KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNN_classifier.fit(X_train, y_train)
# Predicting the Test set results
knn_y_pred = KNN_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, mean_absolute_error

knn_cm = confusion_matrix(y_test, knn_y_pred)
# print(logistic_cm)
knn_ac = (knn_cm[0][0] + knn_cm[1][1]) / (len(y_test))
print(f"The KNN model accuracy {knn_ac * 100} %")
knn_error = mean_absolute_error(y_test, knn_y_pred)
print(f"The KNN model error {knn_error * 100} %")
