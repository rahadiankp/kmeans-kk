import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy

class KMeans(object):
    def __init__(self, dataset, k = 2, keycols = [], epsilon = 0.0001):
        self.dataset = dataset
        self.k = k
        self.keycols = keycols
        self.epsilon = epsilon

    def run(self):
        self.data = self.__generate_data()
        self.centroids = self.__generate_centroids()

        centroids_temp = np.zeros(self.centroids.shape)

        self.clusters = np.zeros(len(self.data))

        error = self.__distance(self.centroids, centroids_temp, None)

        while error >= self.epsilon:
            for idx in range(len(self.data)):
                dist = self.__distance(self.data[idx], self.centroids)
                cluster = np.argmin(dist)
                self.clusters[idx] = cluster

            centroids_temp = deepcopy(self.centroids)
            for idx in range(self.k):
                points = [self.data[jdx] for jdx in range(len(self.data)) if self.clusters[jdx] == idx]
                if not points:
                    ranidx = random.randint(0, len(self.data)-1)
                    points = [self.data[ranidx]]
                    self.clusters[ranidx] = idx
                self.centroids[idx] = np.mean(points, axis=0)
            error = self.__distance(self.centroids, centroids_temp, None)

    def show_clusters(self):
        print(self.clusters)

    def plot(self):
        colors = ['r', 'g', 'b', 'c', 'm', 'k']
        fig, ax = plt.subplots()
        for idx in range(self.k):
            p = np.array([self.data[jdx] for jdx in range(len(self.data)) if self.clusters[jdx] == idx])
            ax.scatter(p[:, 0], p[:, 1], s=15, c=colors[idx])
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=200, c="#505050")
        plt.show()

    def __distance(self, p1, p2, axis = 1):
        return np.linalg.norm(p1-p2, axis=axis)

    def __generate_data(self):
        x = self.dataset[self.keycols[0]].values
        y = self.dataset[self.keycols[1]].values

        return np.array(list(zip(x, y)), dtype=np.float32)

    def __generate_centroids(self):
        centroids_x = np.random.randint(np.min(self.data[:, 0])-self.epsilon-1,
                                        np.max(self.data[:, 0])+self.epsilon+1,
                                        size = self.k
        )
        centroids_y = np.random.randint(np.min(self.data[:, 1])-self.epsilon-1,
                                        np.max(self.data[:, 1])+self.epsilon+1,
                                        size = self.k
        )

        return np.array(list(zip(centroids_x, centroids_y)), dtype=np.float32)

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k

    def set_keycols(self, keycols):
        self.keycols = keycols

    def get_keycols(self):
        return self.keycols


dataset = pd.read_csv('dataset.csv')

a = KMeans(dataset, 5, ['compactness', 'kernel_length'])
a.run()
a.show_clusters()
a.plot()
