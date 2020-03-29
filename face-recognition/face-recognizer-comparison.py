import cv2
import os
import numpy as np
from eigenfaces import read_images, make_train_test_set

#Create face recognizers
def init_model(choice):
    face_recognizer = None
    if choice == "pca":
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
    elif choice == "lda":
        face_recognizer = cv2.face.FisherFaceRecognizer_create()
    elif choice == "lbph":
        face_recognizer =  cv2.face.LBPHFaceRecognizer_create()

    return face_recognizer

def train_model(data, model, folders):
    data_label = {}

    for label in data:
        if label not in data_label:
            data_label[label] = len(data_label)

    train_images, train_labels, test_images, test_labels = make_train_test_set(data, folders)
    train_labels = np.array([data_label[label] for label in train_labels])
    test_labels = np.array([data_label[label] for label in test_labels])

    model.train(train_images, train_labels)

    predictions = cv2.face.StandardCollector_create()

    rank_1_accuracy = 0.0
    top_3_accuracy = 0.0
    top_10_accuracy = 0.0

    for image in range(len(test_images)):
        model.predict_collect(test_images[image], predictions)
        results = predictions.getResults(sorted = True)

        idx = 0

        for (label, dist) in results:
            if idx < 1:
                if label == test_labels[image]:
                    rank_1_accuracy += 1.0
                    top_3_accuracy += 1.0
                    top_10_accuracy += 1.0 
                    break
            elif idx < 3:
                if label == test_labels[image]:
                    top_3_accuracy += 1.0
                    top_10_accuracy += 1.0 
                    break 
            elif idx < 10:
                if label == test_labels[image]:
                    top_10_accuracy += 1.0 
                    break
            else:
                break
            idx += 1

    len_test = len(test_labels)
    rank_1_accuracy = rank_1_accuracy/len_test
    top_3_accuracy = top_3_accuracy/len_test
    top_10_accuracy = top_10_accuracy/len_test

    print("Rank 1 accuracy: {}".format(rank_1_accuracy))
    print("Top 3 accuracy: {}".format(top_3_accuracy))
    print("Top 10 accuracy: {}".format(top_10_accuracy))

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

choice = "lbph"
model = init_model(choice)
train_model(data, model, folders)

# With PCA Face Recognizer
# Rank 1 accuracy: 0.6699029126213593
# Top 3 accuracy: 0.7766990291262136
# Top 10 accuracy: 0.883495145631068

# With LDA Face Recognizer
# Rank 1 accuracy: 0.6699029126213593
# Top 3 accuracy: 0.7087378640776699
# Top 10 accuracy: 0.7669902912621359

# With LBPH Face Recognizer
# Rank 1 accuracy: 0.7864077669902912
# Top 3 accuracy: 0.8543689320388349
# Top 10 accuracy: 0.941747572815534



