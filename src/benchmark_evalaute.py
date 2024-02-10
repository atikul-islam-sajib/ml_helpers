import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from imodels.util.data_util import get_clean_dataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

from classifier import CustomRandomForestClassifier
from regressor import CustomRandomForestRegressor


def evaluate_datasets(datasets, task_type="classification", random_state=42):
    """
    Evaluate datasets using a custom random forest model, either a classifier or a regressor,
    and report the performance scores. This function supports evaluation based on ROC AUC for
    classification tasks and RMSE for regression tasks.

    Parameters:
    -----------
    - `datasets` : list of str
        The names of the datasets to evaluate. These names should correspond to datasets
        accessible via the `get_clean_dataset` function.
    - `task_type` : {'classification', 'regression'}, default='classification'
        Specifies the task type for dataset evaluation. Use 'classification' for ROC AUC score
        evaluation and 'regression' for RMSE.
    - `random_state` : int, default=42
        Controls the randomness of the train/test split and the random forest model construction
        for reproducibility.

    Returns:
    --------
    - `df_scores` : pandas.DataFrame
        A DataFrame containing the evaluation scores for each dataset. It includes columns for
        the dataset name and performance scores using both default and expOOB weighting schemes.

    Examples:
    ---------
    Evaluate classification datasets:

    ```python
    classification_datasets = ["diabetes", "breast_cancer"]
    df_classification_scores = evaluate_datasets(classification_datasets, task_type="classification")
    print(df_classification_scores)
    ```

    Evaluate regression datasets:

    ```python
    regression_datasets = ["boston", "diabetes_reg"]
    df_regression_scores = evaluate_datasets(regression_datasets, task_type="regression")
    print(df_regression_scores)
    ```

    Note:
    -----
    This function relies on `CustomRandomForestClassifier` and `CustomRandomForestRegressor`
    for model fitting and evaluation. Ensure these classes are implemented and configured to
    work with the datasets being evaluated. Additionally, the datasets must be retrievable through
    the `get_clean_dataset` utility function.
    """
    scores_default = []
    scores_expOOB = []

    for dataset_name in datasets:
        # Fetch the dataset
        X, y, feature_names = get_clean_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state
        )

        if task_type == "classification":
            model = CustomRandomForestClassifier(
                oob_score=True, random_state=random_state
            )
        elif task_type == "regression":
            model = CustomRandomForestRegressor(
                oob_score=True, random_state=random_state
            )
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # Initialize and train the model
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        if task_type == "classification":
            # Using ROC AUC for classification
            score_default = roc_auc_score(
                y_test, model.predict_proba(X_test, weights="uniform")[:, 1]
            )
            score_expOOB = roc_auc_score(
                y_test, model.predict_proba(X_test, weights="expOOB")[:, 1]
            )
        elif task_type == "regression":
            # Using RMSE for regression
            score_default = sqrt(
                mean_squared_error(y_test, model.predict(X_test, weights="uniform"))
            )
            score_expOOB = sqrt(
                mean_squared_error(y_test, model.predict(X_test, weights="expOOB"))
            )

        scores_default.append(score_default)
        scores_expOOB.append(score_expOOB)

    # Create a DataFrame
    df_scores = pd.DataFrame(
        {"Dataset": datasets, "Default": scores_default, "expOOB": scores_expOOB}
    )

    return df_scores


if __name__ == "__main__":

    # Define your datasets
    classification_datasets = ["diabetes", "breast_cancer", "heart", "haberman"]
    regression_datasets = ["fico", "enhancer", "credit_g", "juvenile_clean"]

    # Evaluate classification datasets
    df_classification_scores = evaluate_datasets(
        classification_datasets, task_type="classification"
    )

    # Evaluate regression datasetss
    df_regression_scores = evaluate_datasets(
        regression_datasets, task_type="regression"
    )

    # Print scores for each dataset
    print("Classification Scores:")
    print(df_classification_scores)
    print("=" * 100)
    print("\nRegression Scores:")
    print(df_regression_scores)
