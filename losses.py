import numpy as np

eps = 1e-6


def loss_maximum_likelihood(n_r):
    # instead, one might build in a weight matrix cost function, possibly influenced by measurement accuracy and scale
    n_r_reshaped = n_r.reshape(n_r.shape[0], n_r.shape[1], 1, n_r.shape[2])
    covariances = n_r_reshaped.transpose(0, 1, 3, 2) @ n_r_reshaped
    covariances_sum = np.sum(covariances, axis=1)  # sum along t axis
    loss = np.log(np.linalg.det(covariances_sum + eps) + eps)
    return loss - min(loss)


def loss_squared(n_r):
    n_r_reshaped = np.expand_dims(n_r, axis=3)
    covariances = n_r_reshaped.transpose((0, 1, 3, 2,)) @ n_r_reshaped  # maybe add weigth here?
    covariances_sum = np.sum(covariances, axis=(1, 2, 3))  # sum along t axis
    return 0.5 * covariances_sum


def loss_squared_cov(n_r, cov_inv):
    n_r_reshaped = np.expand_dims(n_r, axis=3)
    squared_errors = n_r_reshaped.transpose((0, 1, 3, 2,)) @ cov_inv @ n_r_reshaped  # maybe add weigth here?
    squared_errors_sum = np.sum(squared_errors, axis=(1, 2, 3))  # sum along t axis
    return 0.5 * squared_errors_sum


def loss_angle_sum(n_r):  # performance a lot worse than MLE cost function
    sum_of_theta_errors_squared = np.sum(n_r[:, :, 0] ** 2, axis=1)
    return sum_of_theta_errors_squared
