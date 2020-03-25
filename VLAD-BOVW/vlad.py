import os
import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold   
import cv2

"""
nb_train_samples = 50000 # 3000 training samples
nb_valid_samples = 10000 # 100 validation samples
num_classes = 10

def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize trainging images
    X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
    X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid


(x_train, y_train ,x_test, y_test) = load_cifar10_data(150,150)
print("imported dataset")
"""

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

def create_descriptors(data, data_label):
    descriptor_list = []
    labels_list = []
    for class_label in data:
        for image in data[class_label]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            keypoint, descriptor = desSIFT(gray_image)
            if descriptor is not None:
                descriptor_list.extend(descriptor)
                labels_list.extend([data_label[class_label] for _ in range(len(descriptor))])
    descriptor_list = np.array(descriptor_list)
    labels_list = np.array(labels_list)
    return descriptor_list, labels_list

def getVLADDescriptors(descriptor_list, labels_list, visualDic):
    descriptors = []
    labels = []

    for var in range(len(descriptor_list)):
        # print(desc)
        v = VLAD(descriptor_list[var], visualDic)
        descriptors.append(v)
        labels.append([labels_list[var]])
    
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
data_label = {}

for label in data:
    if label not in data_label:
        data_label[label] = len(data_label)

descriptor_list, labels_list = create_descriptors(data, data_label)
des = descriptor_list.reshape(-1,1)
print(des.shape)
visDic = MiniBatchKMeans(init='k-means++', n_clusters=50,max_iter=1000, batch_size=1000, n_init=10, max_no_improvement=10, verbose=0).fit(des)

X = np.array(descriptor_list).astype(int)
print(descriptor_list[0])
y = labels_list
print(labels_list[0])

cv = StratifiedKFold(n_splits=6, random_state=42)

scores = []
count = 0
for train_ind, validate_ind in cv.split(X, y):
    print("CV # - ", count)
    count += 1
    train_X, train_y = X[train_ind], y[train_ind]
    validate_X, validate_y = X[validate_ind], y[validate_ind]
    
    vlad_des, labels = getVLADDescriptors(train_X, train_y, visDic)
    print ("Hola")

    vlad_des_test, labels_test = getVLADDescriptors(validate_X, validate_y, visDic)
    clf = cv2.ml.KNearest_create()
    clf.train(vlad_des, cv2.ml.ROW_SAMPLE, labels)
    ret, results, neighbours ,dist = clf.findNearest(vlad_des_test, k=10)
    print (results)
    print (labels_test)
