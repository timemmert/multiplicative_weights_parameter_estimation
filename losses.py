import numpy as np

eps = 0.000003


def loss_maximum_likelihood(n_r):
    # instead, one might build in a weight matrix cost function, possibly influenced by measurement accuracy and scale
    n_r_reshaped = n_r.reshape(n_r.shape[0], n_r.shape[1], 1, n_r.shape[2])
    covariances = n_r_reshaped.transpose(0, 1, 3, 2) @ n_r_reshaped
    covariances_sum = np.sum(covariances, axis=1)  # sum along t axis
    return np.log(np.linalg.det(covariances_sum + np.eye(1) * eps))


def loss_angle_sum(n_r):  # performance a lot worse than MLE cost function
    sum_of_theta_errors_squared = np.sum(n_r[:, :, 0] ** 2, axis=1)
    return sum_of_theta_errors_squared
