import numpy as np

from src.svm import evaluate_kernels, train_svms


def main():
    # Random example
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    # Use an additional parameter called heat_kernel with your kerenl function
    # or matrix.
    svms = train_svms(X, y)
    evaluate_kernels(svms, X, y)


if __name__ == "__main__":
    main()
