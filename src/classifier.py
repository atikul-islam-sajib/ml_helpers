import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

sys.path.append("src/")

from utils import generate_unsampled_indices, generate_sample_indices


class CustomRandomForestClassifier(RandomForestClassifier):
    """
    A custom implementation of RandomForestClassifier that supports weighting trees based on their
    out-of-bag (OOB) error.

    This class extends sklearn's RandomForestClassifier, adding functionality to compute and use
    weights for each tree in the ensemble. Weights are derived from the exponential of the negative
    OOB error, enabling more influential contributions from better-performing trees when making predictions.

    Attributes:
    -----------
    - `in_bag_indices_`: list of arrays
        Indices of samples drawn for training each tree.
    - `oob_indices_`: list of arrays
        Out-of-bag sample indices for each tree.
    - `tree_weights_`: list of floats
        Weights for each tree, computed based on their OOB error.

    Methods:
    --------
    - `fit(X, y)`: Fits the random forest model on the input data `X` and target `y`.
    - `predict(X, weights=None)`: Predicts class labels for samples in `X`.
    - `predict_proba(X, weights=None)`: Predicts class probabilities for samples in `X`.
        The `weights` parameter can be either 'uniform' or 'expOOB' to influence prediction.

    Examples:
    ---------
    Uniform weights example:

    ```python
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=42)

    clf = CustomRandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    print(clf.predict(X[:5], weights="uniform"))
    # or
    print(clf.predict_proba(X[:5], weights="uniform"))
    ```

    Exponential OOB weights example:

    ```python
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=42)

    clf = CustomRandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    print(clf.predict(X[:5], weights="expOOB"))
    # or
    print(clf.predict_proba(X[:5], weights="expOOB"))
    ```
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.in_bag_indices_ = []
        self.oob_indices_ = []
        self.tree_weights_ = []

        for estimator in self.estimators_:
            random_state = estimator.random_state
            in_bag_indices = generate_sample_indices(random_state, len(X))
            oob_indices = generate_unsampled_indices(random_state, len(X))

            self.in_bag_indices_.append(in_bag_indices)
            self.oob_indices_.append(oob_indices)

            if len(oob_indices) > 0:
                oob_predictions = estimator.predict(X[oob_indices])
                oob_loss = mean_squared_error(y[oob_indices], oob_predictions)
                self.tree_weights_.append(np.exp(-oob_loss))
            else:
                self.tree_weights_.append(0)

        # Normalize tree weights
        total_weight = np.sum(self.tree_weights_)
        if total_weight > 0:
            self.tree_weights_ = [
                weight / total_weight for weight in self.tree_weights_
            ]

        return self

    def predict(self, X, weights=None):
        """
        Make predictions for the input samples using the custom random forest model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        weights : str, optional (default=None)
            The weighting scheme to use for aggregating tree predictions. Options are:
            - 'expOOB': Use exponential of the negative out-of-bag error as weights.
            - 'uniform': Use uniform weights for all trees.
            If None or an unrecognized string is provided, 'uniform' weighting is used.

        Returns
        -------
        predictions : array of shape (n_samples,)
            The predicted classes.

        Notes
        -----
        The 'expOOB' weighting scheme emphasizes trees with lower OOB errors by assigning them
        higher weights, potentially improving prediction accuracy.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The forest is not fitted yet!")

        weighted_preds = np.zeros((X.shape[0], len(self.classes_)))

        if weights is None or weights not in ["expOOB", "uniform"]:
            weights = "uniform"

        if weights == "expOOB":
            for tree, weight in zip(self.estimators_, self.tree_weights_):
                preds = tree.predict_proba(X)
                weighted_preds += weight * preds
        elif weights == "uniform":
            for tree in self.estimators_:
                preds = tree.predict_proba(X)
                weighted_preds += preds / len(self.estimators_)

        final_preds = np.argmax(weighted_preds, axis=1)
        return self.classes_[final_preds]

    def predict_proba(self, X, weights=None):
        """
        Predict class probabilities for the input samples using the custom random forest model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        weights : str, optional (default=None)
            The weighting scheme to use for aggregating tree predictions. Options are:
            - 'expOOB': Use exponential of the negative out-of-bag error as weights.
            - 'uniform': Use uniform weights for all trees.
            If None or an unrecognized string is provided, 'uniform' weighting is used.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.

        Notes
        -----
        Similar to `predict`, but returns probabilities of each class instead of predicting the class label.
        The 'expOOB' weighting can help in achieving more nuanced probability estimates.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The forest is not fitted yet!")

        weighted_preds = np.zeros((X.shape[0], len(self.classes_)))

        if weights is None or weights not in ["expOOB", "uniform"]:
            weights = "uniform"

        if weights == "expOOB":
            for tree, weight in zip(self.estimators_, self.tree_weights_):
                preds = tree.predict_proba(X)
                weighted_preds += weight * preds
        elif weights == "uniform":
            for tree in self.estimators_:
                preds = tree.predict_proba(X)
                weighted_preds += preds / len(self.estimators_)

        return weighted_preds
