import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import graphviz
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt


class Charts:
    @staticmethod
    def calculate_impurity_reduction(tree):
        """
        Calculate impurity reduction for each node in the decision tree.

        This method computes the decrease in impurity for each node of the tree as a measure of how much each split contributes to organizing the data into homogenous groups.

        Parameters:
        - tree (DecisionTreeClassifier.tree_): The tree object from a fitted DecisionTreeClassifier or a tree from a RandomForestClassifier.

        Returns:
        - numpy.ndarray: An array of impurity reduction values for each node in the tree.
        """
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        impurity = tree.impurity
        weighted_n_node_samples = tree.weighted_n_node_samples

        impurity_reduction = np.zeros(shape=n_nodes)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:  # if not a leaf node
                left_weight = (
                    weighted_n_node_samples[children_left[node]]
                    / weighted_n_node_samples[node]
                )
                right_weight = (
                    weighted_n_node_samples[children_right[node]]
                    / weighted_n_node_samples[node]
                )
                impurity_reduction[node] = impurity[node] - (
                    left_weight * impurity[children_left[node]]
                    + right_weight * impurity[children_right[node]]
                )
        return impurity_reduction

    @staticmethod
    def get_oob_error(model):
        """
        Get Out-Of-Bag (OOB) error for Random Forest models.

        This method retrieves the OOB error from a fitted RandomForestClassifier, which is a measure of prediction error for the training data.

        Parameters:
        - model (RandomForestClassifier): The fitted RandomForestClassifier model.

        Returns:
        - float or None: The OOB error if available, otherwise None.
        """
        if isinstance(model, RandomForestClassifier) and hasattr(model, "oob_score_"):
            return 1 - model.oob_score_
        return None

    @staticmethod
    def get_dot_source(
        tree,
        feature_names,
        class_names,
        impurity_reduction,
        custom_metric,
        oob_error=None,
    ):
        """
        Generate DOT source for tree visualization with node coloring based on impurity.

        This method creates a DOT graph description for visualizing the structure of a decision tree, including details about nodes, their splits, and coloring based on impurity levels.

        Parameters:
        - tree (DecisionTreeClassifier.tree_): The tree object from a fitted model.
        - feature_names (list): List of feature names.
        - class_names (list): List of class names for the target variable.
        - impurity_reduction (numpy.ndarray): Impurity reduction values for each node.
        - custom_metric (numpy.ndarray): An array of custom metric values for each node.
        - oob_error (float, optional): Out-Of-Bag error of the model, if applicable.

        Returns:
        - str: A string representation of the DOT source for the tree visualization.
        """
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        impurity = tree.impurity
        n_node_samples = tree.n_node_samples
        weighted_n_node_samples = tree.weighted_n_node_samples

        # Adjust color settings to match scikit-learn style
        cmap = plt.cm.Blues  # Use Blues color map for impurity-based coloring
        norm = Normalize(
            vmin=0, vmax=max(impurity) if max(impurity) > 0 else 1
        )  # Normalize impurity values
        scalar_mapping = ScalarMappable(norm=norm, cmap=cmap)

        dot_source = "digraph Tree {\n"
        dot_source += 'node [shape=box, style="filled", fontname="helvetica"];\n'
        dot_source += 'edge [fontname="helvetica"];\n'

        if oob_error is not None:
            dot_source += f'label="OOB Error: {oob_error:.2f}\\n";\n'
            dot_source += 'labelloc="t";\n'

        for i in range(n_nodes):
            fillcolor = scalar_mapping.to_rgba(impurity[i], bytes=False)
            hexcolor = to_hex(fillcolor)

            node_info = f"Node {i}\\n"
            node_info += f"Samples: {n_node_samples[i]}\\n"
            node_info += f"Value: {tree.value[i].ravel()}\\n"
            node_info += f"Impurity: {impurity[i]:.2f}\\n"
            node_info += f"Impurity Reduction: {impurity_reduction[i]:.2f}\\n"
            node_info += f"Custom Metric: {custom_metric[i]:.2f}"

            if children_left[i] == children_right[i]:
                dot_source += f'{i} [label="{node_info}", fillcolor="{hexcolor}"];\n'
            else:
                feature_name = feature_names[feature[i]]
                node_info += f"\\nFeature: {feature_name}\\n"
                node_info += f"Threshold: {threshold[i]:.2f}"
                dot_source += f'{i} [label="{node_info}", fillcolor="{hexcolor}"];\n'

            if children_left[i] != -1:
                dot_source += f'{i} -> {children_left[i]} [label="yes"];\n'
            if children_right[i] != -1:
                dot_source += f'{i} -> {children_right[i]} [label="no"];\n'

        dot_source += "}"
        return dot_source

    @staticmethod
    def auto_generate_dot_source(model, feature_names, class_names):
        """
        Automatically generate DOT source for a given tree-based model.

        This method selects the first tree from a RandomForestClassifier or uses the single tree from a DecisionTreeClassifier to generate a DOT source for visualization.

        Parameters:
        - model (DecisionTreeClassifier or RandomForestClassifier): The fitted model.
        - feature_names (list): List of feature names.
        - class_names (list): List of class names for the target variable.

        Returns:
        - str: The DOT source for visualizing the model's first tree.
        """
        is_rf = isinstance(model, RandomForestClassifier)
        tree = model.estimators_[0].tree_ if is_rf else model.tree_
        impurity_reduction = Charts.calculate_impurity_reduction(tree)
        custom_metric = np.random.rand(
            tree.node_count
        )  # Placeholder for a custom metric
        oob_error = Charts.get_oob_error(model) if is_rf else None
        dot_source = Charts.get_dot_source(
            tree,
            feature_names,
            class_names,
            impurity_reduction,
            custom_metric,
            oob_error,
        )
        return dot_source

    @staticmethod
    def plot_tree(model, feature_names, class_names):
        """
        Plot the tree using graphviz.

        This method visualizes the decision tree or the first tree of a random forest using graphviz, highlighting nodes based on their impurity.

        Parameters:
        - model (DecisionTreeClassifier or RandomForestClassifier): The fitted model.
        - feature_names (list): List of feature names.
        - class_names (list): List of class names for the target variable.

        Returns:
        - graphviz.Source: The graphviz object for the tree visualization.
        """
        dot_source = Charts.auto_generate_dot_source(model, feature_names, class_names)
        graph = graphviz.Source(dot_source)
        return graph

    @staticmethod
    def plot_forest_trees(forest_model, feature_names, class_names, max_trees=5):
        """
        Visualize the first few trees in a Random Forest model.

        This method plots up to a specified number of trees from a RandomForestClassifier, showing their structure and metrics like impurity and custom metrics.

        Parameters:
        - forest_model (RandomForestClassifier): The fitted RandomForestClassifier model.
        - feature_names (list): Names of the features.
        - class_names (list): Names of the target classes.
        - max_trees (int): Maximum number of trees to plot. Defaults to 5.

        Note:
        - This method prints the visualization of each tree one by one.
        """
        for idx, estimator in enumerate(forest_model.estimators_[:max_trees]):
            print(f"Visualizing Tree {idx + 1} of {len(forest_model.estimators_)}")
            graph = Charts.plot_tree(estimator, feature_names, class_names)
            display(graph)
