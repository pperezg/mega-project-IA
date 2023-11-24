import numpy as np
import pandas as pd
import os

def verify_directory(path: str):
    """
    Verify if a directory at the specified path exists and create it if not.

    Parameters:
        path (str): The path to the directory.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)

def log_q_function(x: np.array, q: float) -> np.array:
    """
    Compute the logarithmic_q function for an array of values.

    Args:
        x (np.array): Input array of values.
        q (float): q value.

    Returns:
        np.array: Array of transformed values.
    """
    result = np.zeros(x.size)
    for ind in range(x.size):
        if x[ind] <= 0:
            print("Undefined: There is a number less than or equal to 0")
            break
        elif x[ind] > 0 and q == 1:
            result[ind] = np.log(x[ind])
        elif x[ind] > 0 and q != 1:
            result[ind] = (x[ind] ** (1-q) - 1) / (1 - q)
        else:
            print("The conditions were not met")
    return result


def generate_random_values(mu: float, sigma: float, q: float, size: int) -> np.array:
    """
    Generate an array of random values using the Box-Muller transform.

    Args:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        q (float): q value.
        size (int): Number of random values to generate.

    Returns:
        np.array: Array of generated random values.
    """
    U1 = np.random.uniform(0, 1, size)
    U2 = np.random.uniform(0, 1, size)

    q_prime = (1 + q) / (3 - q)
    beta = 1 / (2 * sigma ** 2)

    Z = np.sqrt(-2 * log_q_function(U1, q_prime)) * np.cos(2 * np.pi * U2)

    Z_prime = mu + Z / (np.sqrt(beta * (3 - q)))

    return Z_prime


def generate_random_points(N: int, n: int, mu: float, sigma: float, q: float) -> np.matrix:
    """
    Generate a matrix of random data points.

    Args:
        N (int): Number of rows (data points).
        n (int): Number of columns (features).
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        q (float): q value.

    Returns:
        np.matrix: Matrix of random data points.
    """
    matrix = np.zeros((N, n))
    for i in range(N):
        matrix[i, :] = generate_random_values(mu=mu, sigma=sigma, q=q, size=n)

    return matrix

def calculate_mu_sigma(matrix: np.matrix) -> pd.DataFrame:
    """
    Calculate the mean (mu) and standard deviation (sigma) for each row in the matrix.

    Args:
        matrix (np.matrix): Input matrix of data points.

    Returns:
        pd.DataFrame: DataFrame with 'mu' and 'sigma' columns.
    """
    df = pd.DataFrame({"mu": matrix.mean(axis=1), "sigma": matrix.std(axis=1)})
    return df