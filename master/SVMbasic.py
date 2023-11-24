"""A module to train SVMs and evaluate their kernels."""
from __future__ import annotations
from typing import Callable
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train_svms(
    X: np.ndarray,
    y: np.ndarray,
    heat_kernel: Callable | np.ndarray | None = None,
    random_state: int = 0,
    ) -> SVC:
    """Train an SVM with the given parameters.

    Parameters
    ----------
    X : np.ndarray
        The training data.
    y : np.ndarray
        The training labels.
    heat_kernel : Callable | np.ndarray | None
        The heat kernel. Can be either a matrix or a function.
        By default None.
    random_state : int
        The random state.

    Returns
    -------
    SVC
        The trained SVM.
    """

    # Train the radial basis function SVM.
    svm_rbf = SVC(
        kernel="rbf",
        random_state=random_state,
    )
    svm_rbf.fit(X, y)

    # Train the linear SVM.
    svm_linear = SVC(
        kernel="linear",
        random_state=random_state,
    )
    svm_linear.fit(X, y)

    # Train the polynomial SVM.
    svm_poly = SVC(
        kernel="poly",
        random_state=random_state,
    )
    svm_poly.fit(X, y)

    # Train the heat kernel SVM.
    if callable(heat_kernel):
        svm_heat = SVC(
            kernel=heat_kernel,
            random_state=random_state,
        )
        svm_heat.fit(X, y)
    elif isinstance(heat_kernel, np.ndarray):
        assert heat_kernel.shape == (X.shape[0], X.shape[0])

        svm_heat = SVC(
            kernel="precomputed",
            random_state=random_state,
        )
        svm_heat.fit(heat_kernel, y)
    else:
        svm_heat = None

    return {"rbf": svm_rbf, "linear": svm_linear, "poly": svm_poly, "heat": svm_heat}


def evaluate_kernels(
    svms: dict[str, SVC | None], X_test: np.ndarray, y_test: np.ndarray
    ):
    """
    Evaluate the kernels of the given SVMs. The evaluation is done by
    computing the accuracy score of each SVM on the test set.
    """
    print("============================================")
    for kernel, svm in svms.items():
        if svm is None:
            continue
        y_pred = svm.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        print(
            f"Accuracy Score of {kernel} kernel: {accuracy}",
        )
    print("============================================")