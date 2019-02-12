import numpy as np
import matplotlib.pyplot as plt


def rand_centroids(data_set, k):
    centroids = np.zeros((k, 2))
    for i in range(k):
        index = int(np.random.uniform(0,len(data_set)))
        centroids[i][0] = data_set[index][0]
        centroids[i][1] = data_set[index][1]

    return centroids


def distance(data_set,centroids):
    return [(data_set[0]-centroids[0])*(data_set[0]-centroids[0]) + (data_set[1]-centroids[1])*(data_set[1]-centroids[1])]


def assign_centroids(data_set, centroids):
    for i in range(len(data_set)):
        temp_distance = distance(data_set[i],centroids[0])
        data_set[i][2]=0
        for j in range(len(centroids)):
            if distance(data_set[i],centroids[j])<temp_distance:
                temp_distance = distance(data_set[i],centroids[j])
                data_set[i][2] = j

    return data_set


def reposition_centroids(data_set, centroids):
    x1_sum = np.zeros(len(centroids))
    x2_sum = np.zeros(len(centroids))
    count_per_centorid = np.zeros(len(centroids))

    for i in range(len(data_set)):
        for j in range(len(centroids)):
            if (data_set[i][2] == j):
                x1_sum[j] += data_set[i][0]
                x2_sum[j] += data_set[i][1]
                count_per_centorid[j] += 1

    for k in range(len(centroids)):
        centroids[k][0] = x1_sum[k]/count_per_centorid[k]
        centroids[k][1] = x2_sum[k]/count_per_centorid[k]

    return centroids


def kmeans(data_set, k, max_iterations):
    centroids = rand_centroids(data_set, k)
    iterations = 0

    #old_boys = [] ***FOR CONVERGENCE***

    while iterations < max_iterations:

        data_set = assign_centroids(data_set, centroids)

        centroids = reposition_centroids(data_set, centroids)

        iterations += 1

    return centroids




data_set = np.zeros((1000, 3))
for i in range(len(data_set)):
    data_set[i][0] = int(np.random.uniform(-100, 100))
    data_set[i][1] = int(np.random.uniform(-100, 100))


centroids = kmeans(data_set, 6, 100)
print(centroids)

colors = ['red', 'blue', 'yellow', 'orange','cyan','magenta', 'green', 'purple']


for j in range(len(data_set)):
    plt.scatter(data_set[j][0],data_set[j][1], marker = 'o', color = colors[int(data_set[j][2])])
    #print(data_set[i][2])

for i in range(len(centroids)):
    plt.scatter(centroids[i][0],centroids[i][1], marker = 'o', color = 'black', s = 50)



plt.show()



#DEBUGGING STUFF

#print(data_set)
# x = rand_centroids([(1,1),(2,2),(3,3)],1)
# print(x)

# centroid_counter = np.zeros(len(centroids))
# for i in range(len(data_set)):
#     for j in range(len(centroids)):
#         if (data_set[i][2] == j):
#             centroid_counter[j] += 1
#
# print(centroid_counter)
