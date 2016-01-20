# -*- coding: UTF-8 -*-
import numpy as np
from scipy.linalg import *

# 在algorithm裡自己寫的library
from algorithm.Kmeans import Kmeans

class SpectralClustering(object):

    # Gaussian kernel similarity function
    def gaussian_sim(self,v1, v2):
        sigma = 1
        return np.exp( (-norm(v1-v2)**2)/(2*(sigma**2)) )

    # 建立 W matrix
    def construct_W (self,vecs):
        n = len(vecs)
        W = np.zeros((n, n))
        for i in xrange(n):
            for j in xrange(i,n):
                W[i,j] = W[j,i] = self.gaussian_sim(vecs[i], vecs[j])
        return W

    # 建立降維後的矩陣
    def construct_kdMatrix(self,data,kd):
        W = self.construct_W (data)
        D = np.diag([reduce(lambda x,y:x+y, Wi) for Wi in W])
        L = D - W

        evals, evcts = eig(L,D)
        vals = dict (zip(evals, evcts.transpose()))
        keys = vals.keys()
        keys.sort()
        kdM = np.array ([vals[index] for index in keys[:kd]]).transpose()
        return kdM

    def spectralClustering(self,data,k,kd):
        kdM = self.construct_kdMatrix(data,kd)
        centroids,idx = Kmeans().kMeans(kdM, k)
        return centroids,idx