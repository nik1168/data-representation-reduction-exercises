from __future__ import absolute_import, division, print_function

import random

import sklearn
from sklearn import datasets
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import copy

# Step 1: Load the Digit dataset

digits = datasets.load_digits(n_class=5)
X = digits.data
print(X.shape)


# Step 2: euclidean distance
def euclidean_distances(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    distances = np.sqrt(np.sum(np.square(A[:, np.newaxis] - B), axis=2))
    return distances


def get_vector_without_index(M, index):
    response = []
    for idx, element in enumerate(M):
        if idx != index:
            response.append(element)
    return response


# Step 3: find eps-neighborhood of a point
def find_eps_neighborhood(distances, idx, eps):
    '''
    Input arguments:
        - distances: a matrix containing distances between all pairs of points in the dataset
        - idx: index of the point of interest
        - eps: the epsilon parameter
    Output:
        - Return a set of points in the neighborhood.
    '''
    point_to_evaluate = distances[idx]
    response = set()
    for idx, element in enumerate(point_to_evaluate):
        if element <= eps:
            response.add(idx)
    return response


def test_find_eps_neighborhood():
    distances = np.array([[0, 3, 2], [3, 0, 1], [2, 1, 0]])
    eps = 2
    idx = 0
    return find_eps_neighborhood(distances, idx, eps)


print(test_find_eps_neighborhood())  # should return {0, 2}


# Step 4: find all reachable points of a given point w.r.t eps
def find_reachable_pts(distances, eps, ind):
    eps_neighbors = find_eps_neighborhood(distances, ind, eps)
    reachables = eps_neighbors
    new_pts = copy.deepcopy(eps_neighbors)
    new_pts.remove(ind)
    while len(new_pts) > 0:
        pt = new_pts.pop()
        pt_neighbors = find_eps_neighborhood(distances, pt, eps)
        additional_pts = pt_neighbors.difference(reachables)
        reachables.update(additional_pts)
        new_pts.update(additional_pts)
    return reachables


## Step 5: DBSCAN algorithm
def dbscan(X, eps, minPts):
    ''' a simple implementation of DBSCAN algorithm
    In this implementation, a point is represented by its index in the dataset.
    In this function, except for the step to calculate the Euclidean distance,
    we will only work with the points't indices.

    Input arguments:
        - X: the dataset
        - eps: the epsilon parameter
        - minPts: the minimum number of points for a cluster
    Output:
        - core_points: a list containing the indices of the core points
        - cluster_labels: a Numpy array containing labels for each point in X
        - outliers: a set containing the indices of the outlier points
    '''
    # a list to keep track of the unvisited points
    unvisited = set(range(X.shape[0]))
    # list of core points (or cluster centroids)
    core_points = []
    # list of clusters, each cluster is a set of points
    clusters = []
    # set of outlier points (or noises)
    outliers = set()
    distances = euclidean_distances(X, X)

    while True:
        # randomly choose a point, p, from the list of unvisited points
        p = random.choice(tuple(unvisited))

        # find the eps-neighborhood of the chosen point p
        eps_neighborhood = find_eps_neighborhood(distances, p, eps)

        # check if p is a core point or not### YOUR CODE HERE ###
        is_core_pt = len(eps_neighborhood) >= minPts

        if is_core_pt:
            # add the chosen index to the core_points list
            core_points.append(p)

            # find all reachable points from p w.r.t eps and form a new cluster
            ### YOUR CODE HERE ###
            reachables = find_reachable_pts(distances, eps, p)

            # add the newly formed cluster to the list of cluster
            clusters.append(reachables)

            # remove the indices in the new_cluster from the unvisited set and the outlier set,
            # if they were added to either those set before
            for element in reachables:
                if element in unvisited:
                    unvisited.remove(element)
                if element in outliers:
                    outliers.remove(element)
        else:
            # if not core point, add p to the list of outlier points
            outliers.add(p)

        # remove the chosen index from the unvisited set (if it is still inside this list)
        if p in unvisited:
            unvisited.remove(p)

            # if there is no point left in the unvisited set, stop the loop
        if len(unvisited) == 0:
            break

    # convert the resulting cluster list to cluster_labels
    cluster_labels = np.zeros(X.shape[0])
    for i in range(len(clusters)):
        for j in clusters[i]:
            cluster_labels[j] = i

    return core_points, cluster_labels, outliers


eps = 20.0
minPts = 10
core_points, cluster_labels, outliers = dbscan(X, eps, minPts)
print('%d clusters found' % (len(core_points)))
print('%d outlier points detected' % (len(outliers)))

# visualize the clustering result
selected_cluster = 1
X_cluster_1 = X[cluster_labels == selected_cluster]
n_img_per_row = 10
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img_indx = i * n_img_per_row + j
        if img_indx < len(X_cluster_1):
            img[ix:ix + 8, iy:iy + 8] = X_cluster_1[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection of 100 images from cluster {:}'.format(selected_cluster))
plt.show()

# Calculate the shlhouette score
if len(core_points) > 1:
    print('Silhouette score: %f' % silhouette_score(X, cluster_labels))
else:
    print('Cannot evaluate silhouetter score with only one cluster')
