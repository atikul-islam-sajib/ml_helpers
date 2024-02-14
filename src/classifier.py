import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

sys.path.append("src/")

from utils import generate_unsampled_indices, generate_sample_indices


class CustomRandomForestClassifier(RandomForestClassifier):
    """
    A custom implementation of `RandomForestClassifier` supporting weighting trees based on their out-of-bag (OOB) error.

    This class extends `sklearn.ensemble.RandomForestClassifier`, adding functionality to compute and use weights for
    each tree in the ensemble, derived from the exponential of the negative OOB error. This approach allows for more
    influential contributions from better-performing trees when making predictions.

    Inherits from `sklearn.ensemble.RandomForestClassifier`:
    ```python
    sklearn.ensemble.RandomForestClassifier(
        n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
        bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
        class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None
    )
    ```
        
    Parameters:
        n_estimators : int, default=100
            The number of trees in the forest.
        criterion : {"gini", "entropy", "log_loss"}, default="gini"
            The function to measure the quality of a split.
        max_depth : int, default=None
            The maximum depth of the tree.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
            The number of features to consider when looking for the best split.
        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees.
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization accuracy.
        n_jobs : int, default=None
            The number of jobs to run in parallel for both `fit` and `predict`.
        random_state : int, RandomState instance or None, default=None
            Controls both the randomness of the bootstrapping of the samples used when building trees (if `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node (if `max_features < n_features`).
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
        warm_start : bool, default=False
            When set to `True`, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
        class_weight : dict, list of dict, "balanced", "balanced_subsample" or None, default=None
            Weights associated with classes in the form `{class_label: weight}`.
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.
        max_samples : int or float, default=None
            The number of samples to draw from X to train each base estimator.

    Attributes:
        in_bag_indices_ (List[ndarray]): Indices of samples drawn for training each tree.
        oob_indices_ (List[ndarray]): Out-of-bag sample indices for each tree.
        tree_weights_ (List[float]): Weights for each tree, computed based on their OOB error.

    Methods:
        fit(X, y): Fit the random forest model on the input data `X` and target `y`.
        predict(X, weights=None): Predict class labels for samples in `X`, with an option to use custom tree weights.
        predict_proba(X, weights=None): Predict class probabilities for samples in `X`, with an option to use custom
                                        tree weights.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, n_features=4,
        ...                            n_informative=2, n_redundant=0,
        ...                            random_state=42)
        >>> clf = CustomRandomForestClassifier(n_estimators=10)
        >>> clf.fit(X, y)
        >>> print(clf.predict(X[:5], weights="uniform"))
        >>> print(clf.predict_proba(X[:5], weights="uniform"))

        For exponential OOB weights:
        >>> print(clf.predict(X[:5], weights="expOOB"))
        >>> print(clf.predict_proba(X[:5], weights="expOOB"))
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

        Parameters:
            weights : str, optional (default=None). The weighting scheme to use for aggregating tree predictions. Options are:
            - 'expOOB': Use exponential of the negative out-of-bag error as weights.
            - 'uniform': Use uniform weights for all trees.
            If None or an unrecognized string is provided, 'uniform' weighting is used.

        Returns:
            predictions : array of shape (n_samples,)
            The predicted classes.

        Notes:
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

        Parameters:
            weights : str, optional (default=None). The weighting scheme to use for aggregating tree predictions. Options are:
            -'expOOB': Use exponential of the negative out-of-bag error as weights.
            -'uniform': Use uniform weights for all trees.
            
            If None or an unrecognized string is provided, 'uniform' weighting is used.

        Returns:
            proba : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.

        Notes:
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
