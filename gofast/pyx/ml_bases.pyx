# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# ml_bases.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pow

# Function to calculate the dot product between two vectors
cpdef double dot_product(cnp.ndarray[cnp.float64_t, ndim=1] vec1, cnp.ndarray[cnp.float64_t, ndim=1] vec2):
    """
    Calculate the dot product of two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector.

    vec2 : np.ndarray
        Second vector.

    Returns
    -------
    double
        The dot product of the two vectors.
    """
    if vec1.shape[0] != vec2.shape[0]:
        raise ValueError("Vectors must be the same length.")
    return np.dot(vec1, vec2)

# Function to calculate Euclidean distance between two points
cpdef double euclidean_distance(cnp.ndarray[cnp.float64_t, ndim=1] point1, cnp.ndarray[cnp.float64_t, ndim=1] point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : np.ndarray
        First point.

    point2 : np.ndarray
        Second point.

    Returns
    -------
    double
        The Euclidean distance between the two points.
    """
    cdef double dist = 0.0
    cdef int i
    for i in range(point1.shape[0]):
        dist += pow(point1[i] - point2[i], 2)
    return sqrt(dist)

# Element-wise addition of two arrays
cpdef cnp.ndarray[cnp.float64_t, ndim=1] elementwise_add(cnp.ndarray[cnp.float64_t, ndim=1] arr1, cnp.ndarray[cnp.float64_t, ndim=1] arr2):
    """
    Perform element-wise addition of two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.

    arr2 : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        The element-wise sum of the two arrays.
    """
    return arr1 + arr2

cpdef double weighted_sum(double[:] inputs, double[:] weights, double bias):
    """
    Compute the weighted sum of inputs with a bias term, used in neural networks.

    Parameters
    ----------
    inputs : ndarray
        A 1D numpy array of inputs.
    weights : ndarray
        A 1D numpy array of weights corresponding to the inputs.
    bias : double
        The bias term.

    Returns
    -------
    double
        The weighted sum of the inputs plus the bias.
    """
    cdef int i
    cdef double sum = 0.0
    for i in range(inputs.shape[0]):
        sum += inputs[i] * weights[i]
    sum += bias
    return sum

cpdef double sigmoid(double x):
    """
    Compute the sigmoid activation function.

    Parameters
    ----------
    x : double
        The input value.

    Returns
    -------
    double
        The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

cpdef void gradient_descent_step(double[:] theta, double[:] X, double[:] y, double alpha):
    """
    Perform a single step of gradient descent for linear regression.

    Parameters
    ----------
    theta : ndarray
        The parameter vector (weights).
    X : ndarray
        The feature matrix.
    y : ndarray
        The target vector.
    alpha : double
        The learning rate.
    """
    cdef int m = X.shape[0]
    cdef int n = theta.shape[0]
    cdef int i, j
    cdef double error, grad

    for j in range(n):
        grad = 0
        for i in range(m):
            error = (weighted_sum(X[i, :], theta, 0) - y[i])
            grad += error * X[i, j]
        theta[j] -= (alpha / m) * grad

cpdef double gini_impurity(double[:] labels):
    """
    Compute the Gini impurity of a set of labels.

    Parameters
    ----------
    labels : ndarray
        A 1D numpy array of class labels.

    Returns
    -------
    double
        The Gini impurity of the labels.
    """
    cdef int i
    cdef double impurity = 1.0
    cdef double p
    cdef dict label_counts = {}

    for i in range(labels.shape[0]):
        label_counts[labels[i]] = label_counts.get(labels[i], 0) + 1

    for label in label_counts:
        p = label_counts[label] / labels.shape[0]
        impurity -= p ** 2

    return impurity

cpdef double manhattan_distance(double[:] vec1, double[:] vec2):
    """
    Compute the Manhattan distance between two vectors.

    Parameters
    ----------
    vec1 : ndarray
        A 1D numpy array (first vector).
    vec2 : ndarray
        A 1D numpy array (second vector).

    Returns
    -------
    double
        The Manhattan distance between the two vectors.
    """
    cdef int i
    cdef double sum = 0.0
    for i in range(vec1.shape[0]):
        sum += abs(vec1[i] - vec2[i])
    return sum

cpdef double[:] softmax(double[:] z):
    """
    Compute the softmax function on a vector.

    Parameters
    ----------
    z : ndarray
        A 1D numpy array of input values.

    Returns
    -------
    ndarray
        The output of the softmax function.
    """
    cdef int i
    cdef double max_z = np.max(z)
    cdef double sum_exp = 0.0
    cdef double[:] softmax_vals = np.zeros(z.shape[0], dtype=np.float64)

    for i in range(z.shape[0]):
        softmax_vals[i] = np.exp(z[i] - max_z)
        sum_exp += softmax_vals[i]

    for i in range(z.shape[0]):
        softmax_vals[i] /= sum_exp

    return softmax_vals


cpdef double l2_regularization(double[:] weights, double lambda_):
    """
    Compute the L2 regularization term.

    Parameters
    ----------
    weights : ndarray
        A 1D numpy array of model weights.
    lambda_ : double
        The regularization strength.

    Returns
    -------
    double
        The L2 regularization term.
    """
    cdef int i
    cdef double reg_term = 0.0
    for i in range(weights.shape[0]):
        reg_term += weights[i] ** 2
    return lambda_ * reg_term

cpdef double logistic_loss(double[:] y_true, double[:] y_pred, double lambda_, double[:] weights):
    """
    Compute the logistic loss.

    Parameters
    ----------
    y_true : ndarray
        A 1D numpy array of true labels.
    y_pred : ndarray
        A 1D numpy array of predicted labels.
    lambda_ : double
        Regularization strength.
    weights : ndarray
        A 1D numpy array of model weights.

    Returns
    -------
    double
        The logistic loss.
    """
    cdef int i
    cdef double loss = 0.0
    cdef int n_samples = y_true.shape[0]
    for i in range(n_samples):
        loss += -y_true[i] * np.log(y_pred[i]) - (1 - y_true[i]) * np.log(1 - y_pred[i])
    loss /= n_samples
    loss += l2_regularization(weights, lambda_)
    return loss

cpdef void batch_gradient_descent(double[:] weights, double[:] X, double[:] y, double alpha, int n_iterations):
    """
    Perform batch gradient descent optimization.

    Parameters
    ----------
    weights : ndarray
        A 1D numpy array of model weights.
    X : ndarray
        A 2D numpy array of input features.
    y : ndarray
        A 1D numpy array of target values.
    alpha : double
        The learning rate.
    n_iterations : int
        Number of iterations to run the optimization.
    """
    cdef int i, j, k
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef double[:] gradients = np.zeros(n_features, dtype=np.float64)
    cdef double prediction

    for i in range(n_iterations):
        for j in range(n_features):
            gradients[j] = 0
        for j in range(n_samples):
            prediction = 0
            for k in range(n_features):
                prediction += X[j, k] * weights[k]
            for k in range(n_features):
                gradients[k] += (prediction - y[j]) * X[j, k]
        for j in range(n_features):
            weights[j] -= alpha * gradients[j] / n_samples

cpdef void ridge_regression_solver(double[:] weights, double[:] X, double[:] y, double lambda_, double alpha, int n_iterations):
    """
    Solve Ridge Regression using gradient descent.

    Parameters
    ----------
    weights : ndarray
        A 1D numpy array of model weights.
    X : ndarray
        A 2D numpy array of input features.
    y : ndarray
        A 1D numpy array of target values.
    lambda_ : double
        Regularization strength.
    alpha : double
        The learning rate.
    n_iterations : int
        Number of iterations to run the optimization.
    """
    cdef int i, j, k
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef double[:] gradients = np.zeros(n_features, dtype=np.float64)
    cdef double prediction, reg_term

    for i in range(n_iterations):
        for j in range(n_features):
            gradients[j] = 0
        for j in range(n_samples):
            prediction = 0
            for k in range(n_features):
                prediction += X[j, k] * weights[k]
            for k in range(n_features):
                gradients[k] += (prediction - y[j]) * X[j, k]
        reg_term = 0
        for j in range(n_features):
            reg_term += weights[j]
        for j in range(n_features):
            weights[j] -= alpha * (gradients[j] / n_samples + lambda_ * reg_term)


cpdef void k_means_clustering(double[:] X, double[:] centroids, int[:] labels, int n_clusters, int n_iterations):
    """
    Perform K-Means clustering.

    Parameters
    ----------
    X : ndarray
        A 2D numpy array of input features.
    centroids : ndarray
        A 2D numpy array of initial centroids.
    labels : ndarray
        A 1D numpy array of cluster labels for each sample.
    n_clusters : int
        Number of clusters.
    n_iterations : int
        Number of iterations to run the clustering.
    """
    cdef int i, j, k, cluster
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef double min_dist, dist, diff

    for i in range(n_iterations):
        for j in range(n_samples):
            min_dist = 1e12
            for k in range(n_clusters):
                dist = 0
                for cluster in range(n_features):
                    diff = X[j, cluster] - centroids[k, cluster]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    labels[j] = k
        for k in range(n_clusters):
            for cluster in range(n_features):
                centroids[k, cluster] = 0
            count = 0
            for j in range(n_samples):
                if labels[j] == k:
                    for cluster in range(n_features):
                        centroids[k, cluster] += X[j, cluster]
                    count += 1
            if count > 0:
                for cluster in range(n_features):
                    centroids[k, cluster] /= count

cpdef int[:] k_means_step(double[:,:] data, double[:,:] centroids):
    """
    Perform one step of the K-means clustering algorithm: assigning points to the nearest centroid.

    Parameters
    ----------
    data : ndarray
        2D array where each row is a data point.
    centroids : ndarray
        2D array where each row is a centroid.

    Returns
    -------
    ndarray
        An array of cluster indices for each data point.
    """
    cdef int n_points = data.shape[0]
    cdef int n_centroids = centroids.shape[0]
    cdef int[:] labels = np.empty(n_points, dtype=np.intc)

    cdef int i, j
    cdef double min_dist, dist
    cdef int min_index

    for i in range(n_points):
        min_dist = euclidean_distance(data[i, :], centroids[0, :])
        min_index = 0
        for j in range(1, n_centroids):
            dist = euclidean_distance(data[i, :], centroids[j, :])
            if dist < min_dist:
                min_dist = dist
                min_index = j
        labels[i] = min_index

    return labels

cpdef double split_quality(double[:] left_labels, double[:] right_labels):
    """
    Compute the quality of a split in a decision tree based on Gini impurity.

    Parameters
    ----------
    left_labels : ndarray
        Labels in the left split.
    right_labels : ndarray
        Labels in the right split.

    Returns
    -------
    double
        The computed quality of the split.
    """
    cdef double left_impurity = gini_impurity(left_labels)
    cdef double right_impurity = gini_impurity(right_labels)
    cdef double total = left_labels.shape[0] + right_labels.shape[0]

    return (left_labels.shape[0] / total) * left_impurity + (right_labels.shape[0] / total) * right_impurity

cpdef void sgd_step(double[:] theta, double[:] X, double y, double alpha):
    """
    Perform a single step of Stochastic Gradient Descent (SGD) for linear models.

    Parameters
    ----------
    theta : ndarray
        The parameter vector (weights).
    X : ndarray
        The feature vector.
    y : double
        The target value.
    alpha : double
        The learning rate.
    """
    cdef int n = theta.shape[0]
    cdef int i
    cdef double error = (weighted_sum(X, theta, 0) - y)

    for i in range(n):
        theta[i] -= alpha * error * X[i]

cpdef double svm_margin(double[:] theta, double[:,:] support_vectors):
    """
    Calculate the margin of a Support Vector Machine (SVM).

    Parameters
    ----------
    theta : ndarray
        The parameter vector (weights).
    support_vectors : ndarray
        The support vectors of the SVM.

    Returns
    -------
    double
        The calculated margin.
    """
    cdef int n_vectors = support_vectors.shape[0]
    cdef int i
    cdef double margin = 0.0

    for i in range(n_vectors):
        margin += np.dot(theta, support_vectors[i, :])

    return margin / np.linalg.norm(theta)


