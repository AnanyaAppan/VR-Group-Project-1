import cv2
import numpy as np

#Read images 
# img = cv2.imread('something.jpg')

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def flatten_image_vectors(image):
    """
    Takes in the image as np array of size 256x256 pixels and returns an np array of sixe 1 x (256x256)
    """
    return image.flatten()

def stack_vectors(images):
    stack = []
    for img in images:
        img = flatten_image_vectors(img)
        img = normalized(img,0)
        stack.append(img)

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

def sort_eigenvec_by_eigenval(eigenvals, eigenvecs):
    """
    1. Make a list of (eigenvalue, eigenvector) tuples
    2. Sort the (eigenvalue, eigenvector) tuples from high to low
    """
    eig_pairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in range(len(eigenvals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return  eig_pairs

def choose_k_eigenvecs(eig_pairs, k):
    stack_tuple = (eig_pairs[0][1].reshape(eig_pairs[0][1].shape[0],1),)
    for var in range(1,k):
        stack_tuple += (eig_pairs[var][1].reshape(eig_pairs[var][1].shape[0],1),)

    matrix_w = np.hstack(stack_tuple)
    return matrix_w

def get_transformed_images(matrix_w,stack):
    """
    The dimension of the transformed matrix k x number_of_images  
    """
    transformed = matrix_w.T.dot(stack)
    return transformed.T

def find_closest_image(transformed, image):
    img = normalized(image,0)
    y_hat = matrix_w.T.dot(img)
    argmin = numpy.linalg.norm(y_hat-transformed[0])
    image_id = 0
    for var in range(1, transformed.shape[0]):
        dist = numpy.linalg.norm(y_hat-transformed[var])
        if dist < argmin:
            argmin = dist
            image_id = var

    return image_id
