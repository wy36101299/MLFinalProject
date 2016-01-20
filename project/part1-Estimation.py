# -*- coding: UTF-8 -*-

import numpy as np
import datetime

# 在algorithm裡自己寫的library
from algorithm.GaussianMixtureModel import GaussianMixtureModel

testData_gaussianClusters2D = "../data/testData_gaussianClusters2D.txt"
trainData_gaussianClusters2D = "../data/trainData_gaussianClusters2D.txt"
verificationData_gaussianClusters2D = "../data/verificationData_gaussianClusters2D.txt"

def loaddata():
    x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = [];
    for path in [testData_gaussianClusters2D,trainData_gaussianClusters2D,verificationData_gaussianClusters2D]:
        with open(path, 'r') as f:
            read_data = f.readlines()
        for line in read_data:
            line = line.strip()
            line = line.split(',')
            if line[2] == '1':
                x1.append(float(line[0]))
                y1.append(float(line[1]))
            elif line[2] == '2':
                x2.append(float(line[0]))
                y2.append(float(line[1]))
            elif line[2] == '3':
                x3.append(float(line[0]))
                y3.append(float(line[1]))
        return x1,y1,x2,y2,x3,y3

x1,y1,x2,y2,x3,y3 = loaddata()
data = np.array([[a,b] for a,b in zip(x1+x2+x3,y1+y2+y3)])
start = datetime.datetime.now()
mu,sigma = GaussianMixtureModel().gmm(3,data)
end = datetime.datetime.now()
print "run time"
print end - start
print "mu"
print mu
print "sigma"
print sigma