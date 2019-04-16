'''
TITLE: Program to implement K-Nearest Neighbor Classification Algorithm.

Akshay S. Chavan            BE-B(25)

'''
#import the packages
import pandas as pd
import numpy as np

#Read dataset
dataset=pd.read_csv("C:\\Users\\Akshay Chavan\\PycharmProjects\\Datasets\\KNN-Dataset.csv")

print('---------Dataset Records--------')
print(dataset)

#separating input features from output feature
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values
print('\n\n--------Input Features(X)--------')
print(X)
print('\n\n--------Output Feature(Y)--------')
print(y)
#data visualization

import matplotlib.pyplot as plt

plt.scatter(X[:,0],y,c='r',label='w.r.t. X',s=200,marker='o')
plt.scatter(X[:,1],y,c='b',label='w.r.t. Y',s=100,marker='*')
plt.title('Data Visualization')
plt.xlabel('X/Y Features')
plt.ylabel('Class')
plt.legend()
plt.show()



#import KNeighborshood Classifier and create object of it
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)
print('\n\nAccuracy of general KNN: ',classifier.score(X,y)*100)
#predict the class for the point(6,6)
X_test=np.array([[6,6]])
y_pred=classifier.predict(X_test)
print('Predicted class test point ',X_test[0],' with general KNN: ',y_pred[0])

classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)
print('Accuracy of distance weighted KNN: ',classifier.score(X,y)*100)

#predict the class for the point(6,2)
X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print('Predicted class for test point ',X_test,' with distance weighted KNN: ',y_pred[0])

'''
OUTPUT:

---------Dataset Records--------
   X  Y     Class
0  2  4  negative
1  4  6  negative
2  4  4  positive
3  4  2  negative
4  6  4  negative
5  6  2  positive


--------Input Features(X)--------
[[2 4]
 [4 6]
 [4 4]
 [4 2]
 [6 4]
 [6 2]]


--------Output Feature(Y)--------
['negative' 'negative' 'positive' 'negative' 'negative' 'positive']


Accuracy of general KNN:  33.33333333333333
Predicted class test point  [6 6]  with general KNN:  negative
Accuracy of distance weighted KNN:  100.0
Predicted class for test point  [6 2]  with distance weighted KNN:  positive

'''