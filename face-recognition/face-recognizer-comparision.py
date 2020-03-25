import cv2
import os
import numpy as np
from eigenfaces import read_images, make_train_test_set

#Create face recognizers
# face_recognizer_lbph =  cv2.face.LBPHFaceRecognizer_create()
face_recognizer_ef = cv2.face.EigenFaceRecognizer_create()
# face_recognizer_ff = cv2.face.FisherFaceRecognizer_create()

folders = ["Aadhithya",
           "Abhijith",
           "Abhiramon",
           "Aditya",
           "Akhila",
           "Anagha",
           "Ankush",
           "Anshuk",
           "Deepika",
           "Deepti",
           "Devyani",
           "Harsha_3rd_year",
           "Harsha_5th_year",
           "Hatim",
           "Himank",
           "Juhi",
           "Karthik",
           "Mahesh",
           "Naman",
           "Nehal",
           "Palash",
           "Prachi",
           "Pragya",
           "Pranav",
           "Pranith",
           "Rachit",
           "Rakshith",
           "Ravi",
           "Rohil",
           "SaiPradeep",
           "Shabaz",
           "Shashikant",
           "Shiloni",
           "Shivang",
           "Sowmya",
           "Sravya",
           "Tripti",
           "Utkarsh",
           "Vaibhav",
           "Vamsi",
           "Vishnu",
           "Vivek"                      
          ]
path = "../pre-processing/images/outlinedImages/"

data = read_images(path, folders)
data_label = {}

for label in data:
    if label not in data_label:
        data_label[label] = len(data_label)

train_images, train_labels, test_images, test_labels = make_train_test_set(data, folders)
train_labels = np.array([data_label[label] for label in train_labels])
test_labels = np.array([data_label[label] for label in test_labels])

# face_recognizer_lbph.train(train_images.tolist(), train_labels)
face_recognizer_ef.train(train_images, train_labels)
# face_recognizer_ff.train(train_images, train_labels)

accuracy = 0.0
confidence = 0.0

for image in range(len(test_images)):
    # label1, confidence1 = face_recognizer_lbph.predict(test_images[image])
    label2, confidence2 = face_recognizer_ef.predict(test_images[image])
    # label3, confidence3 = face_recognizer_ff.predict(test_images[image])
    
    if label2 == test_labels[image]:
        accuracy += 1.0
        confidence += confidence2

accuracy = accuracy/(len(test_labels))
confidence = confidence/(len(test_labels))

print("Accuracy: {}".format(accuracy))
print("Confidence: {}".format(confidence))

# With PCA Face Recognizer
# Accuracy: 0.6699029126213593
# Confidence: 4123.371433789167

# With LBPH Face Recognizer
# Accuracy: 0.7864077669902912
# Confidence: 20.911634407443234

# With LDA Face Recognizer
# Accuracy: 0.6699029126213593
# Confidence: 472.9173659286237
