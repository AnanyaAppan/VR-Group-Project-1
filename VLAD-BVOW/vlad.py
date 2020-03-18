#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import cifar10
import numpy as np 
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2


# In[2]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


print(x_train[0].shape)
# print(x_test[0])


def desSIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    #draw keypoints
    #import matplotlib.pyplot as plt		
    #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    #plt.imshow(img2),plt.show()
    return kp,des

def describeORB( image):
    #An efficient alternative to SIFT or SURF
    #doc http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor 
    #with many modifications to enhance the performance
    orb=cv2.ORB_create()
    kp, des=orb.detectAndCompute(image,None)
    return kp,des


# In[60]:


def getDescriptors(images) : 
    descriptors = []
    
    for image in images : 
        print (image.shape)
        kp, des = desSIFT(image)
        if des is not None : 
            descriptors.append(des)
            
    descriptors = list(itertools.chain.from_iterable(descriptors))
    descriptors = np.asarray(descriptors)
        
    return descriptors

def getVLADDescriptors(images, images_lables, visualDic):
    descriptors = []
    labels = []
    
    count = 0
    for image in images : 
        kp, des = desSIFT(image)
        if des is not None : 
            v = VLAD(des, visualDic)
            descriptors.append(v)
            labels.append(images_lables[count])
        count += 1
            
            
    descriptors = list(itertools.chain.from_iterable(descriptors))
    descriptors = np.asarray(descriptors)
        
    return descriptors, labels
    
def kMeans(training, k) : 
    est = KMeans(n_clusters = k, init = 'k-means++').fit(training)
    return est

def VLAD(X, visualDictionary) : 
    
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters
    
    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V


# In[ ]:


sift_des = getDescriptors(np.concatenate((x_train, x_test), axis = 0))
visDic = kMeans(sift_des, 500)


# In[64]:


vlad_des, labels = getVLADDescriptors(x_train, y_train, visDic)
print ("Hola")
vlad_des_test, labels_test = getVLADDescriptors(x_test, y_test, visDic)


# In[ ]:


clf = cv2.ml.KNearest_create()
clf.train(vlad_des, cv2.ml.ROW_SAMPLE, np.asarray(labels, dtype=np.float32))


# In[27]:


ret, results, neighbours ,dist = clf.findNearest(vlad_des_test, k=10)


# In[ ]:


print (results)
print (labels_test)

