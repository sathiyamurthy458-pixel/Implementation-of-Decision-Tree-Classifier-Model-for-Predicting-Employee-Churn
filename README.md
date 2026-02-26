<img width="1200" height="662" alt="image" src="https://github.com/user-attachments/assets/158f0a78-d104-40bc-b36f-4010b2813cb4" /># Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sathiya Murthy k
RegisterNumber: 25006776
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```
## Output:

<img width="1383" height="320" alt="image" src="https://github.com/user-attachments/assets/ed5ad238-e294-4f48-94fe-783bb8f134e4" />

<img width="1035" height="433" alt="image" src="https://github.com/user-attachments/assets/51f6d6f2-f4bd-493c-940f-f8992abbdd10" />

<img width="511" height="313" alt="image" src="https://github.com/user-attachments/assets/45bb5309-a54b-4515-8a2a-289456c2b133" />

<img width="591" height="121" alt="image" src="https://github.com/user-attachments/assets/82e878d2-1935-4f80-b912-807c007d86e0" />

<img width="1379" height="307" alt="image" src="https://github.com/user-attachments/assets/94f358dd-b84c-4c67-bed8-f61b79ffa2a5" />

<img width="1310" height="282" alt="image" src="https://github.com/user-attachments/assets/12a10e4f-89fa-476a-b8ea-e18163ee95ee" />

<img width="568" height="84" alt="image" src="https://github.com/user-attachments/assets/ab96aa9c-1db8-4cde-b2b2-bae0114f34ea" />

<img width="1381" height="165" alt="image" src="https://github.com/user-attachments/assets/00199de2-e8fd-475c-a3dd-cb9fec1e866c" />

<img width="941" height="650" alt="image" src="https://github.com/user-attachments/assets/a734cd34-998e-4ffa-b71c-3341bef91efe" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
