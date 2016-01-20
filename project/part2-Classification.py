# -*- coding: UTF-8 -*-

import pandas as pd
from numpy import *
from scipy.linalg import *
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.lda import LDA

# 在algorithm裡自己寫的library
from algorithm.SpectralClustering import SpectralClustering


# twospirals
trainData_twospirals = "../data/trainData_twospirals.txt"
testData_twospirals = "../data/testData_twospirals.txt"
verificationData_twospirals = "../data/verificationData_twospirals.txt"

# clusterincluster
trainData_clusterincluster = "../data/trainData_clusterincluster.txt"
testData_clusterincluster = "../data/testData_clusterincluster.txt"
verificationData_clusterincluster = "../data/verificationData_clusterincluster.txt"

# gaussianClusters2D
trainData_gaussianClusters2D = "../data/trainData_gaussianClusters2D.txt"
testData_gaussianClusters2D = "../data/testData_gaussianClusters2D.txt"
verificationData_gaussianClusters2D = "../data/verificationData_gaussianClusters2D.txt"

# halfkernel
trainData_halfkernel = "../data/trainData_halfkernel.txt"
testData_halfkernel = "../data/testData_halfkernel.txt"
verificationData_halfkernel = "../data/verificationData_halfkernel.txt"

def loaddata1(path):
    '''
    twospirals , clusterincluster , halfkernel
    '''
    with open(path, 'r') as f:
        read_data = f.readlines()
    x0 = []; y0 = []; x1 = []; y1 = [];
    for line in read_data:
        line = line.strip()
        line = line.split(',')
        if line[2] == '0':
            x0.append(float(line[0]))
            y0.append(float(line[1]))
        elif line[2] == '1':
            x1.append(float(line[0]))
            y1.append(float(line[1]))
    x = x0+x1; y = y0+y1;
    dataset = np.array([[xi,yi]for xi,yi in zip(x,y)])
    label = [0]*len(x0) + [1]*len(x1)
    return x0,y0,x1,y1,dataset,label

def loaddata2(path):
    '''
    gaussianClusters2D
    '''
    x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = [];
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
    x = x1+x2+x3; y = y1+y2+y3;
    dataset = np.array([[xi,yi]for xi,yi in zip(x,y)])
    label = [1]*len(x1) + [2]*len(x2)+ [3]*len(x3)
    return x1,y1,x2,y2,x3,y3,dataset,label

'''
比較所有原始圖形
'''
paths = [trainData_twospirals,testData_twospirals,verificationData_twospirals,
        trainData_clusterincluster,testData_clusterincluster,verificationData_clusterincluster,
        trainData_halfkernel,testData_halfkernel,verificationData_halfkernel,
        trainData_gaussianClusters2D,testData_gaussianClusters2D,verificationData_gaussianClusters2D
       ]

for index in range(len(paths)):
    if index < 9 :
        x0,y0,x1,y1,dataset,label = loaddata1(paths[index])
        print "class : 0"
        # print (x0,y0)
        print "class : 1"
        # print (x1,y1)
    else:
        x1,y1,x2,y2,x3,y3,dataset,label = loaddata2(paths[index])
        print "class : 1"
        # print (x1,y1)
        print "class : 2"
        # print (x2,y2)
        print "class : 3"
        # print (x3,y3)

'''
比較原始圖形,經過spectralClustering,經過lda
'''
test_paths = [testData_twospirals,testData_clusterincluster,testData_halfkernel,testData_gaussianClusters2D]
train_paths = [trainData_twospirals,trainData_clusterincluster,trainData_halfkernel,trainData_gaussianClusters2D]
for index in range(len(train_paths)):
    name = test_paths[index].split('/')
    name = name[2].split('.')
    name = name[0]
    if  name != "testData_gaussianClusters2D":
        # 原本答案
        x0,y0,x1,y1,dataset,label = loaddata1(test_paths[index])
        print "原本答案"
        print "class : 0"
        # print (x0,y0)
        print "class : 1"
        # print (x1,y1)

        # spectralClustering
        centroids,pre_label = SpectralClustering().spectralClustering(dataset,k=2,kd=2)
        x0 = []; y0 = []; x1 = []; y1 = [];
        for data,t_label in zip(dataset,pre_label):
            if t_label == 0:
                x0.append(data[0])
                y0.append(data[1])
            elif t_label == 1:
                x1.append(data[0])
                y1.append(data[1])
        print "spectralClustering"
        print "class : 0"
        # print (x0,y0)
        print "class : 1"
        # print (x1,y1)

        # lda
        x0,y0,x1,y1,train_dataset,train_label = loaddata1(train_paths[index])
        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(train_dataset, train_label)
        x0,y0,x1,y1,test_dataset,test_label = loaddata1(test_paths[index])
        pre_label = clf1.predict(test_dataset)
        x0 = []; y0 = []; x1 = []; y1 = [];
        for data,t_label in zip(test_dataset,pre_label):
            if t_label == 0:
                x0.append(data[0])
                y0.append(data[1])
            elif t_label == 1:
                x1.append(data[0])
                y1.append(data[1])
        print "lda"
        print "class : 0"
        # print (x0,y0)
        print "class : 1"
        # print (x1,y1)
    else:
        # 原本答案
        x1,y1,x2,y2,x3,y3,dataset,label = loaddata2(paths[index])
        print "原本答案"
        print "class : 1"
        # print (x1,y1)
        print "class : 2"
        # print (x2,y2)
        print "class : 3"
        # print (x3,y3)

        # spectralClustering
        centroids,pre_label = SpectralClustering().spectralClustering(dataset,k=3,kd=2)
        x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = [];
        for data,t_label in zip(dataset,pre_label):
            if t_label == 0:
                x1.append(data[0])
                y1.append(data[1])
            elif t_label == 1:
                x2.append(data[0])
                y2.append(data[1])
            elif t_label == 2:
                x3.append(data[0])
                y3.append(data[1])
        print "spectralClustering"
        print "class : 1"
        # print (x1,y1)
        print "class : 2"
        # print (x2,y2)
        print "class : 3"
        # print (x3,y3)

        # lda
        x1,y1,x2,y2,x3,y3,train_dataset,train_label = loaddata2(train_paths[index])
        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(train_dataset, train_label)
        x1,y1,x2,y2,x3,y3,test_dataset,test_label = loaddata2(test_paths[index])
        pre_label = clf1.predict(test_dataset)
        x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = [];
        for data,t_label in zip(test_dataset,pre_label):
            if t_label == 1:
                x1.append(data[0])
                y1.append(data[1])
            elif t_label == 2:
                x2.append(data[0])
                y2.append(data[1])
            elif t_label == 3:
                x3.append(data[0])
                y3.append(data[1])
        print "lda"
        print "class : 1"
        # print (x1,y1)
        print "class : 2"
        # print (x2,y2)
        print "class : 3"
        # print (x3,y3)

'''
比較DBSCAN
'''
# testData_twospirals
test_paths = [testData_twospirals,testData_clusterincluster,testData_halfkernel,testData_gaussianClusters2D]
x0,y0,x1,y1,dataset,label = loaddata1(testData_twospirals)
print "class : 0"
# print (x0,y0)
print "class : 1"
# print (x1,y1)

epslist = [2,2.3,3]
for ep in range(len(epslist)):
    db = DBSCAN(eps=epslist[ep], min_samples=10).fit(dataset)
    pre_label = db.labels_
    x0 = []; y0 = []; x1 = []; y1 = [];
    for data,t_label in zip(dataset,pre_label):
        if t_label == 0:
            x0.append(data[0])
            y0.append(data[1])
        elif t_label == 1:
            x1.append(data[0])
            y1.append(data[1])
    print "eps = %d" %epslist[ep]
    print "class : 0"
    # print (x0,y0)
    print "class : 1"
    # print (x1,y1)

# testData_clusterincluster
x0,y0,x1,y1,dataset,label = loaddata1(testData_clusterincluster)
print "class : 0"
# print (x0,y0)
print "class : 1"
# print (x1,y1)

epslist = [0.5,2,3]
for ep in range(len(epslist)):
    db = DBSCAN(eps=epslist[ep], min_samples=10).fit(dataset)
    pre_label = db.labels_
    x0 = []; y0 = []; x1 = []; y1 = [];
    for data,t_label in zip(dataset,pre_label):
        if t_label == 0:
            x0.append(data[0])
            y0.append(data[1])
        elif t_label == 1:
            x1.append(data[0])
            y1.append(data[1])
    print "eps = %d" %epslist[ep]
    print "class : 0"
    # print (x0,y0)
    print "class : 1"
    # print (x1,y1)

# testData_halfkernel
x0,y0,x1,y1,dataset,label = loaddata1(testData_halfkernel)
print "class : 0"
# print (x0,y0)
print "class : 1"
# print (x1,y1)

epslist = [2,3,4]
for ep in range(len(epslist)):
    db = DBSCAN(eps=epslist[ep], min_samples=10).fit(dataset)
    pre_label = db.labels_
    x0 = []; y0 = []; x1 = []; y1 = [];
    for data,t_label in zip(dataset,pre_label):
        if t_label == 0:
            x0.append(data[0])
            y0.append(data[1])
        elif t_label == 1:
            x1.append(data[0])
            y1.append(data[1])
    print "eps = %d" %epslist[ep]
    print "class : 0"
    # print (x0,y0)
    print "class : 1"
    # print (x1,y1)

# testData_gaussianClusters2D
x1,y1,x2,y2,x3,y3,dataset,label = loaddata2(testData_gaussianClusters2D)
print "class : 1"
# print (x1,y1)
print "class : 2"
# print (x2,y2)
print "class : 3"
# print (x3,y3)

epslist = [0.4,0.6,0.8]
for ep in range(len(epslist)):
    db = DBSCAN(eps=epslist[ep], min_samples=10).fit(dataset)
    pre_label = db.labels_
    x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = [];
    for data,t_label in zip(dataset,pre_label):
        if t_label == 0:
            x1.append(data[0])
            y1.append(data[1])
        elif t_label == 1:
            x2.append(data[0])
            y2.append(data[1])
        elif t_label == 2:
            x3.append(data[0])
            y3.append(data[1])
    print "eps = %d" %epslist[ep]
    print "class : 1"
    # print (x1,y1)
    print "class : 2"
    # print (x2,y2)
    print "class : 3"
    # print (x3,y3)

'''
其他 dataset
'''
trainData = "../data/train.csv"
def loadData(path):
    df = pd.read_csv(path)
    df = df.drop('datetime', axis=1)
    label = df['count'].as_matrix()
    dataset = df.drop('count', axis=1).as_matrix()
    percent25 = np.percentile(label, 25)
    percent50 = np.percentile(label, 50)
    percent75 = np.percentile(label, 75)
    discrete = []
    for l in label:
        if l <= percent25:
            discrete.append(1)
        elif percent25 < l <= percent50:
            discrete.append(2)
        elif percent50 < l <= percent75:
            discrete.append(3)
        else:
            discrete.append(4)
    label = np.array(discrete)
    return dataset,label

df = pd.read_csv(trainData)

# 實驗一
train_dataset,label = loadData(trainData)
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label, test_size=.25)

# SVM kernel="linear"
scores = []
for c in range(1,21):
    clf = SVC(kernel="linear", C=0.025*c).fit(X_train, y_train)
    scores.append((clf.score(X_test, y_test),c))
print "SVM kernel=linear"
print "調整C 0.025*(1-20) - Penalty parameter C of the error term"
print sorted(scores, key=lambda tup: tup[0])[-1]
print "\n"

# SVM kernel="gamma"
scores = []
for p in range(1,9):
    p = float(p)/8.
    clf = SVC(gamma=p, C=p).fit(X_train, y_train)
    scores.append((clf.score(X_test, y_test),p))
print "SVM kernel=gaussian"
print "調整0.125-1 - Penalty parameter C of the error term 和 gamma "
print sorted(scores, key=lambda tup: tup[0])[-1]
print "\n"

# RandomForestClassifier
scores = []
for n in range(5,35,5):
    clf = RandomForestClassifier(max_depth=5, n_estimators=n, max_features=1).fit(X_train, y_train)
    scores.append((clf.score(X_test, y_test),n))
print "RandomForestClassifier"
print "調整樹的數量 5,10,15,20,25,30"
print sorted(scores, key=lambda tup: tup[0])[-1]
print "\n"

# DecisionTreeClassifier
scores = []
for p in ['entropy','gini']:
    clf = tree.DecisionTreeClassifier(criterion = p ).fit(X_train, y_train)
    scores.append((clf.score(X_test, y_test),p))
print "DecisionTreeClassifier"
print "information gain的方式 entropy or gini"
print sorted(scores, key=lambda tup: tup[0])[-1]
print "\n"

# 實驗二
train_dataset,label = loadData(trainData)
Standard_train_dataset = StandardScaler().fit_transform(train_dataset)
MinMax_train_dataset = MinMaxScaler().fit_transform(train_dataset)
datasets = [train_dataset,Standard_train_dataset,MinMax_train_dataset]
nor_names = ["正常","min-max","z-score"]
clfs = [
       SVC(kernel="linear", C=0.025*19),
       SVC(gamma=0.25, C=0.25),
       tree.DecisionTreeClassifier(criterion = 'entropy' ),
       RandomForestClassifier(max_depth=5, n_estimators=25, max_features=1),
      ]
names = ["SVM kernel=linear","SVM kernel=gaussian","DecisionTree","RandomForest"]
for name,c in zip(names,clfs):
    print name
    for dataset,nor_name in zip(datasets,nor_names):
        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=.25)
        clf = c.fit(X_train, y_train)
        print nor_name
        print clf.score(X_test, y_test)

# 實驗三 - PCA
clfs = [
       SVC(kernel="linear", C=0.025*19),
       SVC(gamma=0.25, C=0.25),
       tree.DecisionTreeClassifier(criterion = 'entropy' ),
       RandomForestClassifier(max_depth=5, n_estimators=25, max_features=1),
      ]
names = ["SVM kernel=linear","SVM kernel=gaussian","DecisionTree","RandomForest"]
train_dataset,label = loadData(trainData)
train_dataset = MinMaxScaler().fit_transform(train_dataset)
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label, test_size=.25)
print "PCA"
print "\n"
for name,c in zip(names,clfs):
    print name
    scores = []
    for n in range(5,11):
        pca = PCA(n_components=n)
        pca_test_dataset = pca.fit(X_train).transform(X_test)
        pca_train_dataset = pca.fit(X_train).transform(X_train)
        clf = c.fit(pca_train_dataset, y_train)
        scores.append((clf.score(pca_test_dataset, y_test),n))
    print sorted(scores, key=lambda tup: tup[0])[-1]
    print "--"

# 實驗三 - LDA
clfs = [
       SVC(kernel="linear", C=0.025*19),
       SVC(gamma=0.25, C=0.25),
       tree.DecisionTreeClassifier(criterion = 'entropy' ),
       RandomForestClassifier(max_depth=5, n_estimators=25, max_features=1),
      ]
names = ["SVM kernel=linear","SVM","DecisionTree","RandomForest"]
train_dataset,label = loadData(trainData)
train_dataset = MinMaxScaler().fit_transform(train_dataset)
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label, test_size=.25)
print "LDA"
print "\n"
for name,c in zip(names,clfs):
    print name
    scores = []
    for n in range(5,11):
        lda = LDA(n_components=n)
        lda_train_dataset = lda.fit(X_train, y_train).transform(X_train)
        lda_test_dataset = lda.fit(X_train, y_train).transform(X_test)
        clf = c.fit(lda_train_dataset, y_train)
        scores.append((clf.score(lda_test_dataset, y_test),n))
    print sorted(scores, key=lambda tup: tup[0])[-1]
    print "--"