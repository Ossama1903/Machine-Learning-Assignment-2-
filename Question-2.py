import math
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import combinations

# Filter function used in filter(iterable) method
def filter_data_by_cluster(val, cluster):
    if val == cluster:
        return True

# Importing data
df = pd.read_csv('data.csv')
df = df.drop('CustomerID', axis=1)
label_encoder = preprocessing.LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = np.array(df)

minimum_inertia = math.inf
minimum_cost_centroids = []

print("1- Perform K-Medoid")
print("2- Agglomerative Single-Linkage clustering")

operation = int(input("Enter your input: "))

if operation == 1:
    possible_cluster_sets = list(combinations(df, 2))

    for possible_cluster_set in possible_cluster_sets:
        first_cluster = possible_cluster_set[0]
        second_cluster = possible_cluster_set[1]

        assigned_clusters = [-1 for i in range(len(df))]
        distances = [math.inf for i in range(len(df))]

        distance_cluster_1 = 0
        distance_cluster_2 = 0

        for i in range(len(df)):
            distance_cluster_1 = \
                abs(first_cluster[0] - df[i][0])\
                + abs(first_cluster[1] - df[i][1])\
                + abs(first_cluster[2] - df[i][2])\
                + abs(first_cluster[3] - df[i][3])

            distance_cluster_2 = \
                abs(second_cluster[0] - df[i][0])\
                + abs(second_cluster[1] - df[i][1])\
                + abs(second_cluster[2] - df[i][2])\
                + abs(second_cluster[3] - df[i][3])

            if distance_cluster_1 < distances[i]:
                distances[i] = distance_cluster_1
                assigned_clusters[i] = 0

            if distance_cluster_2 < distances[i]:
                distances[i] = distance_cluster_2
                assigned_clusters[i] = 1

        total_inertia = sum(distances)

        if total_inertia < minimum_inertia:
            minimum_inertia = total_inertia
            minimum_cost_centroids = [first_cluster, second_cluster]
            minimum_cost_assigned_clusters = assigned_clusters

    pca = PCA(n_components=2)
    pca.fit(df)
    reduced_df = pca.transform(df)
    pca1 = PCA(n_components=2)
    pca1.fit(minimum_cost_centroids)
    reduced_centroids = pca1.transform(minimum_cost_centroids)

    print(minimum_inertia)
    print(minimum_cost_centroids)
    print(minimum_cost_assigned_clusters)

    cluster_one_ds = []
    cluster_two_ds = []
    print()
    print()
    for i in range(len(assigned_clusters)):
        if assigned_clusters[i] == 0:
            cluster_one_ds.append(reduced_df[i])
        elif assigned_clusters[i] == 1:
            cluster_two_ds.append(reduced_df[i])

    plt.scatter(x=[cluster_one_ds[i][0] for i in range(len(cluster_one_ds))],
                y=[cluster_one_ds[i][1] for i in range(len(cluster_one_ds))],
                c="red")

    plt.scatter(x=[cluster_two_ds[i][0] for i in range(len(cluster_two_ds))],
                y=[cluster_two_ds[i][1] for i in range(len(cluster_two_ds))],
                c="blue")

    plt.show()
elif operation == 2:
    pass