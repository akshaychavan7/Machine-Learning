'''
TITLE: Program to implement Decsion Tree Classifier.

Akshay S. Chavan            BE-B(25)

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the input csv file
dataset= pd.read_csv('C:\\Users\\Akshay Chavan\\PycharmProjects\\Datasets\\decision-tree-dataset.csv')
print('----------Dataset Records----------')
print(dataset)

# Seperating the input and output features
X = dataset.iloc[:, [1,2,3,4]].values          #removing id column as it's totally irrelevant for prediction
y = dataset.iloc[:, [5]].values                #buys column ie. output column

print('\n\nX before  Label Encoding:')
print(X)
print('\n\nY before  Label Encoding:')
print(y)


# Encoding the categorical features in numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

#encode input features
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])

# Encoding output variable
y = labelencoder_X.fit_transform(y)

print('\n\nX after Label Encoding:')
print(X)
print('\n\nY after Label Encoding:')
print(y)

#Data Visualization
import matplotlib.pyplot as plt

plt.scatter(X[:,0], y, c='r',label='Age',s=200)
plt.scatter(X[:, 1], y, c='g', label='Income',s=150)
plt.scatter(X[:, 2], y, c='b', label='Gender',s=100)
plt.scatter(X[:, 3], y, c='y', label='Marital Status',s=50)
plt.xlabel('Age/Income/Gender/Marital Status')
plt.ylabel('Buys')
plt.title('Visualization of all points')
plt.legend()
plt.show()

# Building a decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X, y)

# Point to be tested
X_test = np.array([['< 21', 'Low', 'Female', 'Married']])   #given in problem statement

# Encoding the test point similar as above
X_test[:,0] = labelencoder_X.fit_transform(X_test[:,0])
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1])
X_test[:,2] = labelencoder_X.fit_transform(X_test[:,2])
X_test[:,3] = labelencoder_X.fit_transform(X_test[:,3])
#print('Xtest')
#print(X_test)


# Predicting the class
y_pred = classifier.predict(X_test)
#print(y_pred)

print("\nThe test point belongs to class: " + str(y_pred[0]))

'''
OUTPUT:

----------Dataset Records----------
    id    age  income  gender marital_status buys
0    1   < 21    High    Male         Single   No
1    2   < 21    High    Male        Married   No
2    3  21-35    High    Male         Single  Yes
3    4    >35  Medium    Male         Single  Yes
4    5    >35     Low  Female         Single  Yes
5    6    >35     Low  Female        Married   No
6    7  21-35     Low  Female        Married  Yes
7    8   < 21  Medium    Male         Single   No
8    9   < 21     Low  Female        Married  Yes
9   10    >35  Medium  Female         Single  Yes
10  11   < 21  Medium  Female        Married  Yes
11  12  21-35  Medium    Male        Married  Yes
12  13  21-35    High  Female         Single  Yes
13  14    >35  Medium    Male        Married   No


X before  Label Encoding:
[['< 21' 'High' 'Male' 'Single']
 ['< 21' 'High' 'Male' 'Married']
 ['21-35' 'High' 'Male' 'Single']
 ['>35' 'Medium' 'Male' 'Single']
 ['>35' 'Low' 'Female' 'Single']
 ['>35' 'Low' 'Female' 'Married']
 ['21-35' 'Low' 'Female' 'Married']
 ['< 21' 'Medium' 'Male' 'Single']
 ['< 21' 'Low' 'Female' 'Married']
 ['>35' 'Medium' 'Female' 'Single']
 ['< 21' 'Medium' 'Female' 'Married']
 ['21-35' 'Medium' 'Male' 'Married']
 ['21-35' 'High' 'Female' 'Single']
 ['>35' 'Medium' 'Male' 'Married']]


Y before  Label Encoding:
[['No']
 ['No']
 ['Yes']
 ['Yes']
 ['Yes']
 ['No']
 ['Yes']
 ['No']
 ['Yes']
 ['Yes']
 ['Yes']
 ['Yes']
 ['Yes']
 ['No']]
C:\Users\Akshay Chavan\PycharmProjects\LP-III Programs\venv\lib\site-packages\sklearn\preprocessing\label.py:235: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


X after Label Encoding:
  y = column_or_1d(y, warn=True)
[[1 0 1 1]
 [1 0 1 0]
 [0 0 1 1]
 [2 2 1 1]
 [2 1 0 1]
 [2 1 0 0]
 [0 1 0 0]
 [1 2 1 1]
 [1 1 0 0]
 [2 2 0 1]
 [1 2 0 0]
 [0 2 1 0]
 [0 0 0 1]
 [2 2 1 0]]


Y after Label Encoding:
[0 0 1 1 1 0 1 0 1 1 1 1 1 0]

The test point belongs to class: 1


'''
