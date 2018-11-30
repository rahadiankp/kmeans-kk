import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy
from math import acos, pi, ceil

class KMeans(object):
    def __init__(self, dataset, k = 2, keycols = [], epsilon = 0.0001, with_output = False):
        self.dataset = dataset
        self.k = k
        self.keycols = keycols
        self.epsilon = epsilon
        self.with_output = with_output

    '''
    run the kmeans to guess clusters
    '''
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

    '''
    plots guessed clusters
    '''    
    def plot(self):
        nrow = ceil((len(self.keycols)*(len(self.keycols)-1))/4)
        plt.figure()
        fig,ax=plt.subplots(nrow, 2, figsize=(25, 90))
        count = 0
        color = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]
        for column in combinations(self.keycols, 2):
            idj = count%2
            idi = count//2
            px = []
            py = []
            for i in range(self.k):
                px += [[self.dataset[column[0]].values[j] for j in range(self.dataset.shape[0]) if self.clusters[j] == i]]
                py += [[self.dataset[column[1]].values[j] for j in range(self.dataset.shape[0]) if self.clusters[j] == i]]
            
            for i in range(self.k):
                ax[idi][idj].scatter(px[i], py[i], c=color[i], label='Cluster_'+str(i+1), s=15)
            ax[idi][idj].set(title=column[0]+' vs. '+column[1], xlabel=column[0], ylabel=column[1])
            ax[idi][idj].legend()
            count += 1
        plt.show()
    
    
    '''
    get sse
    read http://user.engineering.uiowa.edu/~ie_155/lecture/K-means.pdf
    '''
    def sse(self, k = 0):
        if not k:
            k = self.k
        sum = 0
        for c in range(self.k):
            points = [self.data[j] for j in range(len(self.data)) if self.clusters[j] == c]
            for p in points:
                sum += self.__distance(p, self.centroids[c], None)**2
        return sum
    
    '''
    plot multiple sse (distinct k) to get elbow, thus
    optimal k
    '''
    def elbow(self, k_start = 1, k_end = 2):
        # store old val to temp
        k_temp = deepcopy(self.k)
        clusters_temp = deepcopy(self.clusters)
        
        sse = []
        for k in range(k_start, k_end+1):
            self.k = k
            self.run()
            sse += [self.sse(k)]

        points = np.array(list(zip(range(k_start, k_end+1), sse)))
        plt.figure(figsize=(10, 10))
        plt.plot(range(k_start, k_end+1), sse)
        plt.plot(range(k_start, k_end+1), sse, 'b*')
        
        plt.title('Elbow for k = '+str(k_start)+' to '+str(k_end))
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.show()
        
        # restore old val
        self.k = deepcopy(k_temp)
        self.clusters = deepcopy(clusters_temp)
    
    '''
    get silhouette score and plot
    read https://en.wikipedia.org/wiki/Silhouette_(clustering)
    '''
    def silhouette(self):
        cluster = [[] for _ in range(self.k)]
        for i in range(len(self.data)):
            cluster[int(self.clusters[i])].append(self.data[i])
        
        sil_score = 0
        sil_plot = [[] for _ in range(self.k)]
        for i in range(len(self.data)):
            a = np.mean(np.linalg.norm(cluster[int(self.clusters[i])]-self.data[i], axis=1))
            b = 999999999999999999 # INF
            for k in range(self.k):
                if k == int(self.clusters[i]):
                    continue
                b = np.min([b, np.mean(np.linalg.norm(cluster[k]-self.data[i], axis=1))])
                
            sil = (b-a)/np.max([a, b])
            sil_plot[int(self.clusters[i])].append(sil)
            sil_score += sil
            
        for k in range(self.k):
            sil_plot[k].sort(reverse = False)
            
        fig, ax = plt.subplots(figsize=(10, 10))
        color = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]
        bar_width = 1
        prev = 0
        for k in range(self.k):
            index = np.arange(len(sil_plot[k])) + prev
            ax.barh(index, sil_plot[k], bar_width, 
                    color=color[k], label='Cluster_'+str(k+1), antialiased = True
            )
            prev += len(sil_plot[k])
        ax.set_ylabel('Data')
        ax.set_xlabel('Score')
        ax.legend()
        plt.show()
        print('Silhouette score :', sil_score)
        
    '''
    its purpose is to test how well the kmeans determining
    the clusters. this will match real output and guessed
    clusters.
    '''    
    def get_score(self):
        if not self.with_output:
            print('Output column not defined. Cannot get score!')
            return
        
        output_column = list(self.dataset.columns)[-1]
        
        # get clusters name
        freq_table = [{} for _ in range(self.k)]
        clusters_name = pd.unique(self.dataset[output_column].values)
        for i in range(self.dataset.shape[0]):
            x = self.dataset[output_column].values[i]
            if x in freq_table[int(self.clusters[i])]:
                freq_table[int(self.clusters[i])][x] += 1
                continue
            freq_table[int(self.clusters[i])][x] = 1
        trans_table = []
        for i in range(self.k):
            trans_table += [max(freq_table[i], key=freq_table[i].get)]
        
        count = int(self.dataset.shape[0])
        for i in range(self.dataset.shape[0]):
            if trans_table[int(self.clusters[i])] != self.dataset[output_column].values[i]:
                count-=1

        print('Score : '+str((count/self.dataset.shape[0])*100)+'%')

    def __distance(self, p1, p2, axis = 1):
        return np.linalg.norm(p1-p2, axis=axis)

    '''
    this will generate data in numpy array type
    from pandas dataframe
    '''
    def __generate_data(self):
        axis = []
        if not self.keycols:
            self.keycols = list(self.dataset.columns)
            if self.with_output:
                self.keycols = self.keycols[:len(self.keycols)-1]
        for col in self.keycols:
            axis += [self.dataset[col].values]
        
        return np.array(list(zip(*axis)), dtype=np.float32)

    '''
    generates random initial centroids
    '''
    def __generate_centroids(self):
        centroids = []
        for i in range(len(self.keycols)):
            centroids += [
                np.random.randint(
                            np.min(self.data[:, i])-1,
                            np.max(self.data[:, i])+1,
                            size = self.k
                )
            ]
        
        return np.array(list(zip(*centroids)), dtype=np.float32)

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k

    def set_keycols(self, keycols):
        self.keycols = keycols

    def get_keycols(self):
        return self.keycols

dataset = pd.read_csv('dataset.csv')

a = KMeans(dataset, 3, epsilon = 1e-5, with_output = True)
a.run()
#a.show_clusters()
a.elbow(1, 10)
#a.plot()
a.silhouette()
a.get_score()
