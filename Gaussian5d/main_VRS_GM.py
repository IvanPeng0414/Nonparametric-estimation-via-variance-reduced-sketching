import time

import numpy as np

from Gaussian_data import GMMDistributionHighDTwoModes
from tensor_estimate import vrs_prediction


DIMENSION = 5
N_TRAIN = 40000
N_TEST = 100000
N_REPEATS = 1

# Tuning parameters for the VRS tensor basis.
MAX_BASIS_SIZE = 15
LOW_RANK_BASIS_SIZE = 3
LOG_TOLERANCE = 1e-16


def relative_l2_error(predicted, truth):
    """Return ||predicted - truth||_2 / ||truth||_2."""
    return np.linalg.norm(predicted - truth, 2) / np.linalg.norm(truth, 2)


def log_density_gap(predicted, truth, tolerance=LOG_TOLERANCE):
    """Estimate E[log truth] - E[log predicted] on positive predictions."""
    valid_predictions = predicted > tolerance
    if not np.any(valid_predictions):
        return np.nan

    return (
        np.mean(np.log(truth[valid_predictions]))
        - np.mean(np.log(predicted[valid_predictions]))
    )


def evaluate_model(label, tensor_shape, dimension, max_basis_size, x_train, x_test, y_true):
    """Fit one VRS model and print its error, KL-style gap, and runtime."""
    start_time = time.time()

    model = vrs_prediction(tensor_shape, dimension, max_basis_size, x_train)
    y_pred = model.predict(x_test)

    print(f"{label} error", relative_l2_error(y_pred, y_true))
    print(f"{label} KL", log_density_gap(y_pred, y_true))
    print(f"{label} time", time.time() - start_time)

    return y_pred


def run_experiment():
    low_rank_shape = [LOW_RANK_BASIS_SIZE] * DIMENSION
    low_rank_shape[0] = MAX_BASIS_SIZE
    full_shape = [MAX_BASIS_SIZE] * DIMENSION

    print("basis sizes:", MAX_BASIS_SIZE, LOW_RANK_BASIS_SIZE)
    print("low-rank tensor shape:", low_rank_shape)
    print("full tensor shape:", full_shape)

    distribution = GMMDistributionHighDTwoModes(
        n_dims=DIMENSION,
        normal_type="Known",
    )

    for repeat in range(N_REPEATS):
        print(f"\nrepeat {repeat + 1}/{N_REPEATS}")

        x_train = distribution.generate_gmm_samples(num_samples=N_TRAIN)
        x_test = distribution.generate_gmm_samples(num_samples=N_TEST)
        y_true = distribution.gmm_pdf_normal(x_test)

        evaluate_model(
            "VRS",
            low_rank_shape,
            DIMENSION,
            MAX_BASIS_SIZE,
            x_train,
            x_test,
            y_true,
        )

        evaluate_model(
            "Tucker full",
            full_shape,
            DIMENSION,
            MAX_BASIS_SIZE,
            x_train,
            x_test,
            y_true,
        )


if __name__ == "__main__":
    run_experiment()
