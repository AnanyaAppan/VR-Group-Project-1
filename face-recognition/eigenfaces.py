import os
import numpy as np
import cv2

def read_images(path, folders):
    data = {}
    for folder in folders:
        images = []
        for filename in os.listdir(path+folder):
            image = cv2.imread(os.path.join(path+folder, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
        images = np.array(images)
        data[folder] = images
    return data

def make_train_test_set(data, folders):
    train_labels = []
    train_images = []
    
    test_labels = []
    test_images = []
    
    for image_label in folders:
        thresh = min(len(data[image_label]),IMG_THRESHOLD)
        for img in data[image_label]:
            if thresh <= 0:
                test_labels.append(image_label)
                test_images.append(img)
            else:
                train_labels.append(image_label)
                train_images.append(img)
            thresh -= 1
                
    #Size of train_images is 563 x 1
    train_images = np.array(train_images)
    #Size of test_images is 103 x 1
    test_images = np.array(test_images)
                
    return train_images, train_labels, test_images, test_labels

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def flatten_image_vectors(image):
    """
    Takes in the image as np array of size 256x256 pixels and returns an np array of sixe 1 x (256x256)
    """
    #Resize each image to 256 x 256 (sanity check)
    image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA) 
    return image.flatten()

def stack_vectors(images):
    stack = []
    for img in images:
        #Size of the image is 65536 x 1
        img = flatten_image_vectors(img)
        #Convert size of the image to 65536 x 1
        img = normalized(img,0).T

        stack.append(np.array(img))
    #Size of the stack is (563, 65536)
    stack = np.squeeze(np.array(stack),axis=2)
    return stack

def compute_mean_vector(stack):
    mean_vector = []
    for var in range(stack.shape[0]):
        mean = np.mean(stack[var, :])
        mean_vector.append(mean)
    
    return np.array(mean_vector)

def compute_scatter_matrix(stack, mean_vector):
    scatter_matrix = np.zeros((mean_vector.shape[0],mean_vector.shape[0]))
    for var in range(stack.shape[1]):
        scatter_matrix += (stack[:,var].reshape(mean_vector.shape[0],1) - mean_vector).dot((stack[:,var].reshape(mean_vector.shape[0],1) - mean_vector).T)

    return scatter_matrix

def compute_covariance_matrix(stack):
    cov_matrix = []
    for var in range(stack.shape[0]):
        cov_matrix.append(stack[var, :])

    return np.cov(cov_matrix)

def get_eigenvecs_vals(matrix):
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    return eigenvals, eigenvecs

def sort_eigenvec_by_eigenval(eigenvals, eigenvecs):
    """
    1. Make a list of (eigenvalue, eigenvector) tuples
    2. Sort the (eigenvalue, eigenvector) tuples from high to low
    """
    eig_pairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in range(len(eigenvals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return  eig_pairs

def choose_k_eigenvecs(eig_pairs, k, image_stack):
    stack_tuple = (eig_pairs[0][1].reshape(eig_pairs[0][1].shape[0],1),)
    for var in range(1,k):
        stack_tuple += (eig_pairs[var][1].reshape(eig_pairs[var][1].shape[0],1),)

    matrix_w = np.hstack(stack_tuple)
    # eig_pairs[var][1].shape[0] : 563 (563 when IMG_THRESHOLD = 15)
    for var in range(image_stack.shape[1]-eig_pairs[var][1].shape[0]):
        matrix_w = np.append(matrix_w, np.array([np.zeros(k)]), axis=0)
    
    # Shape of the matrix W is (65536, 100) (when k = 100)
    print("Shape of the matrix W is {}".format(matrix_w.shape))
    return matrix_w

def get_transformed_images(matrix_w,stack):
    """
    The dimension of the transformed matrix k x number_of_images  
    """
    # Shape of the transformed matrix is (100, 563) (when k = 100)
    transformed = matrix_w.T.dot(stack)
    print("Shape of the transformed matrix is {}".format(transformed.shape))
    return transformed.T

def find_closest_image(transformed, matrix_w, image):
    img = flatten_image_vectors(image)
    img = normalized(img,0).T
    y_hat = matrix_w.T.dot(img)
    y_hat = np.squeeze(y_hat,axis=1)
    argmin = np.linalg.norm(y_hat-transformed[0])
    image_id = 0
    for var in range(1, transformed.shape[0]):
        dist = np.linalg.norm(y_hat-transformed[var])
        if dist < argmin:
            argmin = dist
            image_id = var

    return image_id

if __name__ == "__main__":
    IMG_THRESHOLD = 15
    k = 100

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

    train_images, train_labels, test_images, test_labels = make_train_test_set(data, folders)

    image_stack = stack_vectors(train_images)
    mean_vector = compute_mean_vector(image_stack)
    scatter_matrix = compute_scatter_matrix(image_stack,mean_vector)
    cov_matrix = compute_covariance_matrix(image_stack)

    # eigenvectors and eigenvalues for the from the scatter matrix
    eig_val_sc, eig_vec_sc = get_eigenvecs_vals(scatter_matrix)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = get_eigenvecs_vals(cov_matrix)

    eig_pairs1 = sort_eigenvec_by_eigenval(eig_val_sc, eig_vec_sc)
    eig_pairs2 = sort_eigenvec_by_eigenval(eig_val_cov, eig_vec_cov)

    matrix_w_1 = choose_k_eigenvecs(eig_pairs1, k, image_stack)
    matrix_w_2 = choose_k_eigenvecs(eig_pairs2, k, image_stack)

    transformed1 = get_transformed_images(matrix_w_1, image_stack.T)
    transformed2 = get_transformed_images(matrix_w_2, image_stack.T)

    scatter_matrix_accuracy = 0.0
    covariance_matrix_accuracy = 0.0

    for var in range(len(test_images)):
        img_id1 = find_closest_image(transformed1, matrix_w_1, test_images[var])
        img_id2 = find_closest_image(transformed2, matrix_w_2, test_images[var])
        
        if train_labels[img_id1] == test_labels[var]:
            scatter_matrix_accuracy += 1.0
        if train_labels[img_id2] == test_labels[var]:
            covariance_matrix_accuracy += 1.0
    scatter_matrix_accuracy = scatter_matrix_accuracy/(len(test_labels))
    covariance_matrix_accuracy = covariance_matrix_accuracy/(len(test_labels))

    print("Accuracy with scatter matrix eigen vectors: {}".format(scatter_matrix_accuracy))
    print("Accuracy with covariance matrix eigen vectors: {}".format(covariance_matrix_accuracy))

# With k = 100 and IMG_THRESHOLD = 15:
# Accuracy with scatter matrix eigen vectors: 0.5339805825242718
# Accuracy with covariance matrix eigen vectors: 0.5339805825242718
