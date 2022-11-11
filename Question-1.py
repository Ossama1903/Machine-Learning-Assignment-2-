import math
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


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
centroid_domain = []
inertia = []
centroids = []

print("1- Draw inertia chat against number of clusters")
print("2- Draw clusters")

operation = int(input("Enter your input: "))

if operation == 1:

    for i in range(10):
        centroid_domain.append(df[random.randint(0, 199)])

    for total_iterations in range(1, 9):
        centroids = []

        for i in range(total_iterations):
            centroids.append(centroid_domain[i])

        distances = [math.inf for i in range(len(df))]
        assigned_clusters = [-1 for i in range(len(df))]

        CHANGE = True

        while CHANGE:

            CHANGE = True
            for i in range(0, len(df)):
                for index, centroid in enumerate(centroids):
                    gender_distance = abs(df[i][0] - centroid[0])
                    age_distance = abs(df[i][1] - centroid[1])
                    income_distance = abs(df[i][2] - centroid[2])
                    score_distance = abs(df[i][3] - centroid[3])
                    total_distance = gender_distance + age_distance + income_distance + score_distance

                    if total_distance < distances[i]:
                        distances[i] = total_distance
                        assigned_clusters[i] = index
                        CHANGE = False

            gender_sum = [0 for i in range(len(centroids))]
            age_sum = [0 for i in range(len(centroids))]
            income_sum = [0 for i in range(len(centroids))]
            score_sum = [0 for i in range(len(centroids))]

            for index, assigned_cluster in enumerate(assigned_clusters):
                gender_sum[assigned_cluster] += df[index][0]
                age_sum[assigned_cluster] += df[index][1]
                income_sum[assigned_cluster] += df[index][2]
                score_sum[assigned_cluster] += df[index][3]

            for i in range(len(centroids)):
                datapoint_count = len(list(filter(lambda seq: filter_data_by_cluster(seq, i), assigned_clusters)))
                new_gender = gender_sum[i] / datapoint_count
                new_age = age_sum[i] / datapoint_count
                new_income = income_sum[i] / datapoint_count
                new_score = score_sum[i] / datapoint_count
                centroids[i] = [new_gender, new_age, new_income, new_score]

        inertia.append((len(centroids), sum(distances)))

    plt.plot([inertia[x][0] for x in range(len(inertia))], [inertia[y][1] for y in range(len(inertia))])
    plt.title('Inertia chart')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia / Distance')
    plt.show()

elif operation == 2:
    for i in range(5):
        centroids.append(df[random.randint(0, 199)])
        distances = [math.inf for i in range(len(df))]
        assigned_clusters = [-1 for i in range(len(df))]

        CHANGE = True

        while CHANGE:

            CHANGE = True
            for i in range(0, len(df)):
                for index, centroid in enumerate(centroids):
                    gender_distance = abs(df[i][0] - centroid[0])
                    age_distance = abs(df[i][1] - centroid[1])
                    income_distance = abs(df[i][2] - centroid[2])
                    score_distance = abs(df[i][3] - centroid[3])
                    total_distance = gender_distance + age_distance + income_distance + score_distance

                    if total_distance < distances[i]:
                        distances[i] = total_distance
                        assigned_clusters[i] = index
                        CHANGE = False

            gender_sum = [0 for i in range(len(centroids))]
            age_sum = [0 for i in range(len(centroids))]
            income_sum = [0 for i in range(len(centroids))]
            score_sum = [0 for i in range(len(centroids))]

            for index, assigned_cluster in enumerate(assigned_clusters):
                gender_sum[assigned_cluster] += df[index][0]
                age_sum[assigned_cluster] += df[index][1]
                income_sum[assigned_cluster] += df[index][2]
                score_sum[assigned_cluster] += df[index][3]

            for i in range(len(centroids)):
                datapoint_count = len(list(filter(lambda seq: filter_data_by_cluster(seq, i), assigned_clusters)))
                new_gender = gender_sum[i] / datapoint_count
                new_age = age_sum[i] / datapoint_count
                new_income = income_sum[i] / datapoint_count
                new_score = score_sum[i] / datapoint_count
                centroids[i] = [new_gender, new_age, new_income, new_score]


    pca = PCA(n_components=2)
    pca.fit(df)
    reduced_df = pca.transform(df)
    pca1 = PCA(n_components=2)
    pca1.fit(centroids)
    reduced_centroids = pca1.transform(centroids)

    plt.scatter(x=[reduced_centroids[i][0] for i in range(len(reduced_centroids))],
                y=[reduced_centroids[i][1] for i in range(len(reduced_centroids))],
                c="red",
                zorder=2,
                marker="*")

    cluster_one_ds = []
    cluster_two_ds = []
    cluster_three_ds = []
    cluster_four_ds = []
    cluster_five_ds = []

    for i in range(len(assigned_clusters)):
        if assigned_clusters[i] == 0:
            cluster_one_ds.append(reduced_df[i])
        if assigned_clusters[i] == 1:
            cluster_two_ds.append(reduced_df[i])
        if assigned_clusters[i] == 2:
            cluster_three_ds.append(reduced_df[i])
        if assigned_clusters[i] == 3:
            cluster_four_ds.append(reduced_df[i])
        if assigned_clusters[i] == 4:
            cluster_five_ds.append(reduced_df[i])

    print(cluster_one_ds)

    plt.scatter(x=[cluster_one_ds[i][0] for i in range(len(cluster_one_ds))],
                y=[cluster_one_ds[i][1] for i in range(len(cluster_one_ds))],
                c="green")
    plt.scatter(x=[cluster_two_ds[i][0] for i in range(len(cluster_two_ds))],
                y=[cluster_two_ds[i][1] for i in range(len(cluster_two_ds))],
                c="blue")
    plt.scatter(x=[cluster_three_ds[i][0] for i in range(len(cluster_three_ds))],
                y=[cluster_three_ds[i][1] for i in range(len(cluster_three_ds))],
                c="yellow")
    plt.scatter(x=[cluster_four_ds[i][0] for i in range(len(cluster_four_ds))],
                y=[cluster_four_ds[i][1] for i in range(len(cluster_four_ds))],
                c="pink")
    plt.scatter(x=[cluster_five_ds[i][0] for i in range(len(cluster_five_ds))],
                y=[cluster_five_ds[i][1] for i in range(len(cluster_five_ds))],
                c="orange")

    plt.show()
