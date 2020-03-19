import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold   
from sklearn.neighbors import BallTree

def desSIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    return keypoints, descriptors

def describeORB(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image,None)
    return keypoints, descriptors

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def read_images(path, folders):
    data = {}
    for folder in folders:
        images = []
        for filename in os.listdir(path+folder):
            image = cv2.imread(os.path.join(path+folder, filename))
            if image is not None:
                images.append(image)
        images = np.array(images)
        data[folder] = images
    return data

def create_descriptors(data):
    descriptor_list = []
    image_descriptor = {}
    for class_label in data:
        print(class_label)
        class_descriptor_list = []
        for image in data[class_label]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            keypoint, descriptor = desSIFT(gray_image)
            if descriptor is not None:
                descriptor_list.extend(descriptor)
                class_descriptor_list.append(descriptor)
        class_descriptor_list = np.array(class_descriptor_list)
        image_descriptor[class_label] = class_descriptor_list
    descriptor_list = np.array(descriptor_list)
    return descriptor_list, image_descriptor

def getVLADDescriptors(image_descriptor, data_label, visualDic):
    descriptor = []
    labels = []

    # print(image_descriptor)
    for class_label in image_descriptor:
        print(class_label)
        for desc in image_descriptor[class_label]:
            # print(desc)
            v = VLAD(desc, visualDic)
            descriptors.append(v)
            labels.append([data_label[class_label]])
    
    descriptors = np.array(descriptors)
    labels = np.array(labels).astype(np.float32)

    return descriptors, labels

def VLAD(X, visualDictionary): 
    
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters
    
    m,d = X.shape
    V = np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V

data = read_images("../../../Assignment2/Panorama-BOVW/SIFT-SURF/",["Bikes", "Horses"])
# data = read_images("../../../Assignment2/Panorama-BOVW/SIFT-SURF/cifar-10/",["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
print(data)
descriptor_list, image_descriptor = create_descriptors(data)

visDic = KMeans(descriptor_list, 50)

data_label = {}

for label in data:
    if label not in data_label:
        data_label[label] = len(data_label)

X = []
y = []

for class_label in data:
    for image in data[class_label]:
        X.append(image)
        y.append(class_label)

X = np.asarray(X)
y = np.asarray(y)

cv = StratifiedKFold(n_splits=6, random_state=42)

scores = []
count = 0
for train_ind, validate_ind in cv.split(X, y):
    print("CV # - ", count)
    count += 1
    train_X, train_y = X[train_ind], y[train_ind]
    validate_X, validate_y = X[validate_ind], y[validate_ind]
    descriptor_list, image_descriptor = create_descriptors(train_X)
    vlad_des, labels = getVLADDescriptors(image_descriptor, data_label, visDic)
    print ("Hola")
    descriptor_list, image_descriptor = create_descriptors(validate_X)
    vlad_des_test, labels_test = getVLADDescriptors(image_descriptor, data_label, visDic)
    clf = cv2.ml.KNearest_create()
    clf.train(vlad_des, cv2.ml.ROW_SAMPLE, labels)
    ret, results, neighbours ,dist = clf.findNearest(vlad_des_test, k=10)
    print (results)
    print (labels_test)
    # score = md.score(validate_X, validate_y)
    # scores.append(score)

# sc = np.array(scores)
# print(sc)
# print("Score: " + str(np.mean(sc)))


