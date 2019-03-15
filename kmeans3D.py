import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def rand_centroids(data_set, k):
    centroids = np.zeros((k, 3))
    for i in range(k):
        index = int(np.random.uniform(0, len(data_set)))
        centroids[i][0] = data_set[index][0]
        centroids[i][1] = data_set[index][1]
        centroids[i][2] = data_set[index][2]

    return centroids


def distance2(data_set, centroids): #NOT VALID FOR 3D
    return [(data_set[0]-centroids[0])*(data_set[0]-centroids[0]) + (data_set[1]-centroids[1])*(data_set[1]-centroids[1])]


def distance(data_set, centroids):
    b = [0, 0, 0, 0, 0]
    b[0] = (data_set[0]-centroids[0])*(data_set[0]-centroids[0])
    b[1] = 2*(data_set[0]-centroids[0])*(data_set[1]-centroids[1])
    b[2] = (data_set[1]-centroids[1])*(data_set[1]-centroids[1]) + 2*(data_set[0]-centroids[0])*(data_set[2]-centroids[2])
    b[3] = 2*(data_set[1]-centroids[1])*(data_set[2]-centroids[2])
    b[4] = (data_set[2]-centroids[2])*(data_set[2]-centroids[2])
    total_sum = 0
    t0 = 0
    t1 = 1
    for i in range(len(b)):
        total_sum += b[i]*((t1**(i+1))-(t0**(i+1)))/(i+1)

    return total_sum


def assign_centroids(data_set, centroids):
    for i in range(len(data_set)):
        temp_distance = distance(data_set[i],centroids[0])
        data_set[i][3]=0
        for j in range(len(centroids)):
            if distance(data_set[i],centroids[j]) < temp_distance:
                temp_distance = distance(data_set[i], centroids[j])
                data_set[i][3] = j

    return data_set


def reposition_centroids(data_set, centroids):
    x1_sum = np.zeros(len(centroids))
    x2_sum = np.zeros(len(centroids))
    x3_sum = np.zeros(len(centroids))
    count_per_centorid = np.zeros(len(centroids))

    for i in range(len(data_set)):
        for j in range(len(centroids)):
            if (data_set[i][3] == j):
                x1_sum[j] += data_set[i][0]
                x2_sum[j] += data_set[i][1]
                x3_sum[j] += data_set[i][2]
                count_per_centorid[j] += 1

    for k in range(len(centroids)):
        centroids[k][0] = x1_sum[k]/count_per_centorid[k]
        centroids[k][1] = x2_sum[k]/count_per_centorid[k]
        centroids[k][2] = x3_sum[k]/count_per_centorid[k]

    return centroids

def reposition_centroids2(data_set, centroids):
    dist = np.zeros(len(centroids))
    q_vals = dist
    prev_dist = dist

    for t in range(len(prev_dist)):
        prev_dist[t] = 9999
        q_vals[t] = 9999

    for x in np.arange(-10, 10, 0.1):
        for y in np.arange(-10, 10, 0.1):
            for z in np.arange(-10, 10, 0.1):
                q = [x,y,z]
                dist[dist!=0] = 0
                for i in range(len(data_set)):
                    for j in range(len(centroids)):
                        if (data_set[i][3] == j):
                            dist[j] += distance(data_set[i], q)
                for l in range(len(centroids)):
                    if(dist[l]<prev_dist[l]): #prev dist is a list for some reason rn
                        prev_dist[l] = dist[l]
                        q_vals[l] = q



    return q_vals

def kmeans(data_set, k, max_iterations):
    centroids = rand_centroids(data_set, k)
    iterations = 0

    #old_boys = [] ***FOR CONVERGENCE***

    while iterations < max_iterations:

        data_set = assign_centroids(data_set, centroids)

        centroids = reposition_centroids(data_set, centroids)

        iterations += 1

    return centroids


def function_analysis(data_point,range):
    return (data_point[2])*(range**2) + (data_point[1])*range + data_point[0]


def create_Q(n):
    Q = np.zeros(n,n)
    for i in range(n):
        for j in range(n):
            Q[i][j] = (t**(i+j+1))/(i+j+1)


def avg_distortion(data_set,centroids):
    centroid_counter = np.zeros(len(centroids), 1)
    distortion = 0
    for i in range(len(data_set)):
        for j in range(len(centroids)):
            if data_set[i][3]==j:
                centroid_counter[j]+=1

    for k in range(len(data_set)):
            distortion += (1/centroid_counter[data_set[k][3]])*


data_set = np.zeros((1000, 4))

for i in range(len(data_set)):
    data_set[i][0] = int(np.random.uniform(-100, 100))
    data_set[i][1] = int(np.random.uniform(-100, 100))
    data_set[i][2] = int(np.random.uniform(-100, 100))

centroids = kmeans(data_set, 8, 100)
rangex = np.linspace(0,1,100)

fig = plt.figure()
#ax = plt.axes(projection='3d')
colors = ['red', 'blue', 'yellow', 'orange', 'cyan', 'magenta', 'green', 'purple']

lets_get_messy = []
centroid_combo = []

# for k in range(len(data_set)):
#     lets_get_messy.append(function_analysis(data_set[k], rangex))
#     plt.plot(rangex,lets_get_messy[k],color = colors[int(data_set[k][3])])

for l in range(len(centroids)):
    centroid_combo.append((function_analysis(centroids[l], rangex)))
    plt.plot(rangex, centroid_combo[l], color=colors[l])

#
# for j in range(len(data_set)):
#     ax.scatter3D(data_set[j][0],data_set[j][1], data_set[j][2], marker = 'o', color = colors[int(data_set[j][3])])
#
# for i in range(len(centroids)):
#     ax.scatter3D(centroids[i][0],centroids[i][1], centroids[i][2], marker = 'o', color = 'black', s = 50)

#print(data_set)

#plt.plot(rangex,lets_get_messy)
plt.show()

# UNCOMMENT THE CODE BELOW TO SEE THE COORDINATES OF THE CENTROIDS
# AS WELL AS HOW MANY POINTS BELONG TO THAT CENTROID

# print(centroids)

# centroid_counter = np.zeros(len(centroids))
# for i in range(len(data_set)):
#     for j in range(len(centroids)):
#         if (data_set[i][2] == j):
#             centroid_counter[j] += 1

# print(centroid_counter)



