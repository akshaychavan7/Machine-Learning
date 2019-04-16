'''
TITLE: Program to implement K-means clustering algorithm.

Akshay S. Chavan            BE-B(25)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading the csv file
dataset = pd.read_csv("C:\\Users\\Akshay Chavan\\PycharmProjects\\Datasets\\kmeansdata.csv")
print('----------Dataset Records----------')
print(dataset)

# Selecting the features in variable X
X = dataset.iloc[:, [0,1]].values           #As K-means is an unsupervised algorithm it doesn't have any output feature column
#print(X)

# Initialise the centroids and test point as per the problem statement
init_centroids = np.array([[0.1,0.6], [0.3,0.2]])
test_point_p6 = np.array([0.25, 0.5])

from sklearn.cluster import KMeans

# Kmeans
kmeans = KMeans(n_clusters = 2, init = init_centroids, n_init = 1)
#fit_predict is just a combination of fit() and predict()
y_kmeans = kmeans.fit_predict(X)        # fit_predict will return a list containing classes of all points
print("\n\nPredicted clusters: ", y_kmeans)


# Data Visualization - Optional part
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c='r', label = 'Cluster 0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c='b', label = 'Cluster 1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'green', label = 'Centroids', marker='*')
plt.title('Clusters of Points')
plt.xlabel('X co-ordinate')
plt.ylabel('Y co-ordinate')
plt.legend()
plt.show()


# Finding class of test point
index_P6 = np.where(X == test_point_p6) # where function returns index of test_point_p6 in X
print("\nindex p6", index_P6)
class_P6 = y_kmeans[index_P6[0][0]]     # getting class from y_kmeans

#y_pred=kmeans.predict([test_point_p6])
#print('Y_pred:',y_pred)

# Answer to questions in the problem statement
print("\n\n1. Point P6 belongs to class: ", str(class_P6))
print("2. Population of cluster around m2 is: ", np.count_nonzero(y_kmeans==1))     #number 1 indicates cluster m2
print("3. Updated value of m1 and m2 is: ", str(kmeans.cluster_centers_))

'''
OUTPUT:

----------Dataset Records----------
      X     Y
0  0.10  0.60
1  0.15  0.71
2  0.08  0.90
3  0.16  0.85
4  0.20  0.30
5  0.25  0.50
6  0.24  0.10
7  0.30  0.20


Predicted clusters:  [0 0 0 0 1 0 1 1]

index p6 (array([5, 5], dtype=int64), array([0, 1], dtype=int64))


1. Point P6 belongs to class:  0
2. Population of cluster around m2 is:  3
3. Updated value of m1 and m2 is:  [[0.148      0.712     ]
 [0.24666667 0.2       ]]

'''