# -*- coding: UTF-8 -*-
import numpy as np

class Kmeans(object):
    def kMeans(self,X, K, maxIters = 100):

        N = len(X)
        init_SSE = 100000

        # 選出較好的初始點，預設10次
        for i in range(10):
            random_centroids = X[np.random.choice(np.arange(N), K), :]
            new_init_SSE = 0
            for k in range(K):
                new_init_SSE += sum(np.sum((np.tile(random_centroids[k],(N,1)) - X)*(np.tile(random_centroids[k],(N,1)) - X), axis=1))

            if (new_init_SSE < init_SSE):
                centroids = random_centroids
                init_SSE = new_init_SSE

        D = [ np.sum((np.tile(centroids[k],(N,1)) - X)*(np.tile(centroids[k],(N,1)) - X), axis=1) for k in range(K) ]
        C = np.argmin( np.dstack(tuple(D))[0] , axis=1 )
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        SSE = sum([sum(a) for a in D])

        i = 0
        threshold = 0.001
        while i < maxIters:
            D = [ np.sum((np.tile(centroids[k],(N,1)) - X)*(np.tile(centroids[k],(N,1)) - X), axis=1) for k in range(K) ]
            C = np.argmin( np.dstack(tuple(D))[0] , axis=1 )
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
            new_SSE = sum([sum(a) for a in D])
            if abs(new_SSE - SSE) < threshold:
                break
            SSE = new_SSE
            i += 1
        return np.array(centroids) , list(C)