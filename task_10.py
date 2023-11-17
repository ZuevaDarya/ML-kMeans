import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data_table = pd.read_csv('input_task_10.csv', delimiter=',', decimal='.', index_col='Object')
data = pd.read_csv('input_task_10.csv').values

X = data[:, 1:3]
Y = data[:, 3]

model = KMeans(n_clusters=3, init=np.array([[10.4, 10.8], [14.67, 12.67], [10.0, 7.29]]), max_iter=100, n_init=1)
model.fit(X)

Y_pred = model.predict(X)
print(f'Предсказанные классы: {Y_pred}')

#Центроид для объектов с кдассом 0
center = model.cluster_centers_[0]

#Выбираем объекты с классом 0 из предсказанных данных 
X_class_0 = []

for i in range(len(Y_pred)):
    if Y_pred[i] == 0:
        X_class_0.append(X[i])

#Рассчитываем расстояние
distances = [] 
for x in X_class_0:
    dist = ((center[0] - x[0])**2 + (center[1] - x[1])**2)**0.5
    distances.append(dist)

average_dist = np.average(distances)
print(f'среднее расстояний между объектами и центроидом, отнесенных к кластеру 0: {round(float(average_dist), 3)}')