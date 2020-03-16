import cv2
import numpy as np

#Read images 
img = cv2.imread('something.jpg')

def flatten_image_vectors(image):
    """
    Takes in the image as np array of size 256x256 pixels and returns an np array of sixe 1 x (256x256)
    """
    return image.flatten()

def stack_vectors(images):
    stack = []
    for img in images:
        stack.append(flatten_image_vectors(img))

    return np.array(stack)

def compute_mean_vector(stack):
    mean_vector = []
    for var in range(stack.shape[1]):
        mean = np.mean(stack[var, :])
        mean_vector.append(mean)
    
    return mean_vector

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

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)