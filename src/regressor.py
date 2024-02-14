import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, mean_squared_error

sys.path.append("src/")

from utils import generate_sample_indices, generate_unsampled_indices


class CustomRandomForestRegressor(RandomForestRegressor):
    """
    A custom implementation of RandomForestRegressor that supports weighting trees based on their
    out-of-bag (OOB) error.

    This class extends sklearn's RandomForestRegressor, adding the functionality to compute and use
    weights for each tree in the ensemble. Weights are derived from the exponential of the negative
    OOB error, enabling more influential contributions from better-performing trees when making predictions.
    
    Inherits from `sklearn.ensemble.RandomForestClassifier`:
    ```python
    class sklearn.ensemble.RandomForestRegressor(
        n_estimators=100, *, criterion='squared_error', max_depth=None,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
        oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0,
        max_samples=None, monotonic_cst=None)
    ```


    Parameters:
    -----------
    All parameters of the sklearn.ensemble.RandomForestRegressor class are accepted.
    
    Parameters:
        n_estimators : int, default=100
            The number of trees in the forest.
        criterion : {“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default="“squared_error”"
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
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.
        max_samples : int or float, default=None
            The number of samples to draw from X to train each base estimator.

    Attributes:
    -----------
    - `in_bag_indices_` : list of arrays
        Indices of samples drawn for training each tree.
    - `oob_indices_` : list of arrays
        Out-of-bag sample indices for each tree.
    - `tree_weights_` : list of floats
        Weights for each tree, computed based on their OOB error.
        

    Methods:
    --------
    - `fit(X, y)`: Fits the random forest regressor model on the input data `X` and target `y`.
    - `predict(X, weights=None)`: Predicts regression target for `X`. The `weights` parameter can
    be either 'uniform' or 'expOOB' to influence prediction.

    Examples:
    ---------
    ```python
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=4, n_informative=2, noise=0.5, random_state=42)
    reg = CustomRandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)
    y_pred = reg.predict(X[:5])
    ```
    """

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
            X : array-like or sparse matrix of shape = [n_samples, n_features]
                The training input samples.
            y : array-like of shape = [n_samples] or [n_samples, n_outputs]
                The target values (real numbers).

        Returns: 
            self : object
            Returns self.
        """
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
        Predict regression target for X using the trained forest.

        Parameters:
            weights : {'uniform', 'expOOB'}, default='uniform'
                The weighting scheme to use for aggregating predictions from the individual trees.
                - 'uniform': All trees contribute equally to the final prediction.
                - 'expOOB': Trees are weighted based on the exponential of the negative out-of-bag error.

        Returns:
            y : ndarray of shape = [n_samples]. The predicted values.

        Notes:
            If 'expOOB' weighting is used, trees with lower out-of-bag error have a greater influence on
            the final prediction, potentially improving predictive performance on unseen data.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The forest is not fitted yet!")

        if weights is None or weights not in ["expOOB", "uniform"]:
            weights = "uniform"

        # Collect predictions from each tree
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        if weights == "expOOB":
            # Use the exponential of the negative out-of-bag error as weights
            weighted_preds = np.average(all_preds, axis=0, weights=self.tree_weights_)
        elif weights == "uniform":
            # All trees have equal weight
            weighted_preds = np.mean(all_preds, axis=0)

        return weighted_preds
