import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot

iris = pandas.read_csv('datasets/iris.csv',sep=',')
print(iris.head())

#Removendo a coluna das classes das flores Ex:setosa
X = iris.iloc[:, 0:4].values

#Scikit-Learn fornece 3 modos kmeans++, random
#Parametros init,n_clusters,max_iter,n_jobs, algorithm
kmeans = KMeans(n_clusters=3, init='random')
#Computa o metodo de clustering k-means .
kmeans.fit(X)
#Metodo para ver os centroides
kmeans.cluster_centers_
#Retorna uma tabela de distancias(medidas de similaridade)
kmeans.fit_transform(X)
#tabela de centroides, mostra no atual momento a quais clusters pertencem os pontos
kmeans.labels_

#Metodo Elbow
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print(i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
pyplot.subplot(2,1,1)    
pyplot.plot(range(1, 11), wcss)
pyplot.title('O Metodo Elbow')
pyplot.xlabel('Numero de Clusters')
pyplot.ylabel('WSS') #within cluster sum of squares

#No ponto 3 que seria o “cotovelo”, ou seja, a partir desse ponto 
#que não existe uma discrepância tão significativa em termos de variância.

pyplot.subplot(2,1,2)
pyplot.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
pyplot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
pyplot.title('Iris Clusters and Centroids')
pyplot.xlabel('SepalLength')
pyplot.ylabel('SepalWidth')
pyplot.legend()

pyplot.show()