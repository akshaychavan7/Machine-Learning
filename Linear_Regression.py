'''
TITLE: Assignment on Linear Regression

Akshay S. Chavan            BE-B(25)
'''
import matplotlib.pyplot as plt
import pandas as pd

# Read Dataset
dataset=pd.read_csv("C:\\Users\\Akshay Chavan\\PycharmProjects\\Datasets\\linear-reg-dataset.csv")

print('-------------Dataset Records------------')
print(dataset)

#separate input fields(columns) from ouput field(Y)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#print(X)
# Import the Linear Regression and Create object of it
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
Accuracy=regressor.score(X, y)*100
print("Accuracy :")
print(Accuracy)

# Take user input
hours=int(input('Enter the no of hours: '))

y_pred=regressor.predict([[hours]])
print('Predicted risk score for ',hours,' hours: ',y_pred[0])

print('Equation for line of best fit: Y= ',regressor.coef_[0],'X + ',regressor.intercept_)

#Visualization of data points and line of best fit
plt.scatter(X,y,c='r',label='Datapoints')
plt.title('Data Visualization')
plt.plot(X,regressor.predict(X));
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

'''
OUTPUT:

-------------Dataset Records------------
   no. of hours spent on driving (x)   risk score on scale of 0-100 (y)
0                                  10                                95
1                                   9                                80
2                                   2                                10
3                                  15                                50
4                                  10                                45
5                                  16                                98
6                                  11                                38
7                                  16                                93
Accuracy :
43.709481451010035
Enter the no of hours: 10
Predicted risk score for  10  hours:  58.4636140637776
Equation for line of best fit: Y=  4.587898609975469 X +  12.584627964022907

Graph
'''