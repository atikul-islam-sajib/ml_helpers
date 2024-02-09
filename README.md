# ML Helpers: Enhanced RandomForest with Custom Features

ML Helpers offers advanced versions of the RandomForestClassifier and RandomForestRegressor from scikit-learn, incorporating significant enhancements for improved performance and insights. These custom implementations provide a more flexible and insightful approach to RandomForest modeling.

## Key Enhancements

### Weighted Predictions Based on Out-of-Bag (OOB) Loss

Our CustomRandomForest models introduce weighted predictions, leveraging OOB loss to enhance prediction accuracy:

- **OOB Loss Calculation:** For each tree, the OOB loss is calculated as the mean squared error between actual and predicted OOB samples.
- **Weight Assignment:** This OOB loss is then transformed into a weight for each tree using an exponential function: `exp(-OOB_loss)`. Higher weights are assigned to trees with lower OOB loss, indicating better performance.
- **Weighted Predictions:** During prediction, these weights are used to aggregate outputs from all trees, giving more influence to better-performing trees.

### Access to In-Bag and Out-of-Bag Data

- **Data Tracking:** The algorithm tracks which samples are used for training each tree (in-bag) and which are left out (OOB), offering detailed insights into the model's training process.
- **Analysis and Insights:** Post-training, access to in-bag and OOB samples for each tree is available, facilitating custom analyses and deeper understanding of each tree's interaction with the data.

These features make our CustomRandomForestClassifier and CustomRandomForestRegressor uniquely capable of providing weighted predictions based on tree performance and offering valuable insights into the training data used for each tree.

## Enhanced Tree Visualization

The `plot_tree` functionality has been enhanced to include new features that allow for a more detailed and informative visualization of decision trees within the RandomForest models. This enhancement aids in the interpretability and analysis of the models, providing users with a deeper understanding of how decisions are made within the forest.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/atikul-islam-sajib/ml_helpers.git
cd ml_helpers
pip install -r requirements.txt
```

## Usage

### Classification with CustomRandomForestClassifier

```python
from src.classifier import CustomRandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, random_state=42)

clf = CustomRandomForestClassifier(n_estimators=10)
clf.fit(X, y)
# Example of using weighted predictions
print(clf.predict(X[:5], weights="expOOB"))
```

### Regression with CustomRandomForestRegressor

```python
from src.regressor import CustomRandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=4, n_informative=2, random_state=42)

reg = CustomRandomForestRegressor(n_estimators=100)
reg.fit(X, y)
# Example of using weighted predictions
print(reg.predict(X[:5], weights="expOOB"))
```

### Benchmark Evaluation

Evaluate the models on benchmark datasets:

```python
from src.benchmark_evaluate import evaluate_datasets

classification_scores = evaluate_datasets(["diabetes", "breast_cancer", "heart", "haberman"], task_type="classification")
regression_scores = evaluate_datasets(["fico", "enhancer", "credit_g", "juvenile_clean"], task_type="regression")

print("Classification Scores:", classification_scores)
print("Regression Scores:", regression_scores)
```

```markdown
### Enhanced Tree Visualization

Our `plot_tree` method has been enhanced to support a broader range of models. For Example,

- CustomRandomForestClassifier
- CustomRandomForestRegressor
- scikit-learn's Decision Tree (DT)
- scikit-learn's RandomForest (RF)

This ensures broad applicability for model analysis and interpretation, facilitating a deeper understanding of model behavior across various tree-based modeling approaches.
```

#### Classification Visualization with CustomRandomForestClassifier

```python
from src.chart import Charts
from imodels.util.data_util import get_clean_dataset

X, y, feature_names = get_clean_dataset('heart')
model = CustomRandomForestClassifier(n_estimators=3, max_depth=3)
charts = Charts()
graph = charts.plot_tree(model, feature_names, y)
graph
```

````

#### Regression Visualization with CustomRandomForestRegressor

```python
X, y, feature_names = get_clean_dataset('credit_g')
model = CustomRandomForestRegressor(n_estimators=3, max_depth=3)
graph = charts.plot_tree(model, feature_names, y)
graph
```

#### Regression Visualization with Scikit-Learn DT

```python
X, y, feature_names = get_clean_dataset('credit_g')
model = DecisionTreeRegressor(n_estimators=3, max_depth=3)
graph = charts.plot_tree(model, feature_names, y)
graph
```

This visualization functionality is not limited to our custom models but also extends to the default Decision Tree and RandomForest models provided by scikit-learn. This makes it a versatile tool for anyone working with tree-based models, offering insights into how models make decisions, regardless of the specific implementation used.

#### Compatibility Note

```markdown
The `plot_tree` visualization method is compatible with:

- CustomRandomForestClassifier
- CustomRandomForestRegressor
- scikit-learn's Decision Tree (DT)
- scikit-learn's RandomForest (RF)

This ensures broad applicability for model analysis and interpretation, facilitating a deeper understanding of model behavior across various tree-based modeling approaches.
```

This additional information ensures that users are fully aware of the capabilities and compatibility of the `plot_tree` method, emphasizing its utility not just for custom models but also for standard models from scikit-learn. It highlights the tool's versatility in visualizing and analyzing decision-making processes within tree-based models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We extend our gratitude to the scikit-learn contributors for their foundational work, upon which our custom implementations and enhancements are built.
````
