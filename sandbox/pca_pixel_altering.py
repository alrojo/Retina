import numpy as np


def pca_color_pertub(im):
    """
    This functions adds multiples of the principal components of the covariance matrix
    for RBG pixel values. The intuition behind this distortion of the images is to make
    the object identity invariant to changes in the intenisty and color of the illumination.
    See Krizhevsky et. al 2012 for a more in depth explanation of the method.

    Note that it is assumed that the depth of the picture is the first dimension in image
    array i.e. image.shape = (depth, height or width, height or width).
    """

    # Calculate covariance matrix
    n = im.shape[1]*im.shape[2]
    A = im.reshape([im.shape[0], n])
    A_mean = np.mean(A, axis=1, keepdims = True)
    Q = A - A_mean
    cov = 1.0/(n-1) * np.dot(Q, Q.T)

    # Get eigenvalues and eigenvectors
    eigVal, eigVec = np.linalg.eig(cov)

    # Add multiples of the found principal components to each RBG pixel
    alpha = np.random.normal(0, 0.1, im.shape[0])
    color_distort = np.dot(eigVec, np.expand_dims(alpha*eigVal, axis=1))
    im_distorted = A + color_distort
    im_distorted = im_distorted.reshape(im.shape)

    return im_distorted