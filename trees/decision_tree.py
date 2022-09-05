import numpy as np
import sys


####################
# Training functions
####################
def gini(data):
    score = 1 - sum([(np.count_nonzero(data == label) / np.size(data))**2 for label in np.unique(data)])
    return score


def coefficient_of_variation(data):
    std = np.std(data)
    avg = np.average(data)

    if avg > 0:
        cv = std / avg
        return cv
    else:
        cv = 0
        return cv


def split(data, condition):
    """Split array based on column condition."""
    return [data[condition], data[~condition]]


def weighted_gini(data, splitting_column_index, threshold):
    """Calculate the weighted gini score of a split."""
    split_arrays = split(
        data=data,
        condition=data[:, splitting_column_index] < threshold
    )

    score = sum([(np.size(split) / np.size(data)) * gini(split[:, -1]) for split in split_arrays])
    return score


def standard_deviation_reduction(data, splitting_column_index, threshold):
    """Calculate the standard deviation reduction of a split."""
    split_arrays = split(
        data=data,
        condition=data[:, splitting_column_index] < threshold
    )

    overall_std = np.std(data[:, -1])
    split_stds = []
    for split_array in split_arrays:
        if split_array.shape[0] > 0:
            split_std = (np.size(split_array) / np.size(data)) * np.std(split_array[:, -1])
            split_stds.append(split_std)
        else:
            split_stds.append(0)

    score = overall_std - sum(split_stds)
    return score


def determine_best_split(data, split_type='c'):
    """Determine the best split for the data.

    For classification, split the data by every threshold on every column.
    Returns the column and threshold which result in the minimum weighted gini
    score.

    For regression, split the data by every threshold on every column.
    Returns the column and threshold which result in the maximum standard
    deviation reduction.

    args:
        data (numpy array): data to determine the best split for.
        split_type (str)  : specifies whether to determine the best split for
                            classification ('c') or regression ('r').

    returns:
        best_split (tuple): tuple indicating the index of the best splitting
                            column, the best threshold for splitting, and the
                            best gini or standard deviation reduction score.
    """

    if split_type == 'c':
        each_split_best_gini_scores = []
        for splitting_column_index in range(0, data.shape[1] - 1):
            threshold_scores = []
            thresholds = np.unique(data[:, splitting_column_index])
            for threshold in thresholds:
                weighted_gini_score = weighted_gini(
                    data=data,
                    splitting_column_index=splitting_column_index,
                    threshold=threshold
                )
                threshold_scores.append((threshold, weighted_gini_score))

            best_threshold, best_gini = min(threshold_scores, key=lambda t: t[1])
            each_split_best_gini_scores.append((splitting_column_index, best_threshold, best_gini))

        best_split = min(each_split_best_gini_scores, key=lambda t: t[2])
        return best_split

    elif split_type == 'r':
        each_split_best_std_reduction_scores = []
        for splitting_column_index in range(0, data.shape[1] - 1):
            threshold_scores = []
            thresholds = np.unique(data[:, splitting_column_index])
            for threshold in thresholds:
                std_reduction = standard_deviation_reduction(
                    data=data,
                    splitting_column_index=splitting_column_index,
                    threshold=threshold
                )
                threshold_scores.append((threshold, std_reduction))

            best_threshold, best_std_reduction = max(threshold_scores, key=lambda t: t[1])
            each_split_best_std_reduction_scores.append((splitting_column_index, best_threshold, best_std_reduction))

        best_split = max(each_split_best_std_reduction_scores, key=lambda t: t[2])
        return best_split

    else:
        print("Error: split type (classification 'c' or regression 'r') must be specified")
        sys.exit()


###########
# Classifer
###########
class Tree():
    """Base tree class."""
    def __init__(self, vertices=[], edges=[]):
        self.vertices = vertices
        self.edges = edges

    def add_vertices(self, vertices):
        for vertex in vertices:
            self.vertices.append(vertex)

    def add_edges(self, edges):
        for edge in edges:
            self.edges.append(edge)

    def remove_vertices(self, vertices):
        for vertex in vertices:
            self.edges.remove(vertex)

    def remove_edges(self, edges):
        for edge in edges:
            self.edges.remove(edge)


class DecisionTreeClassifier(Tree):
    """Decision tree classifier"""
    def fit(self, x_train, y_train, tol=0.0):
        data = np.concatenate((x_train, y_train), axis=1)

        stack = [(0, data)]
        while len(stack) > 0:
            index, data = stack.pop(0)

            splitting_column_index, threshold, best_gini = determine_best_split(data)

            is_leaf = False
            if gini(data[:, -1]) <= tol:
                is_leaf = True

            # Assign labels to leaves.
            label = None
            if is_leaf:
                # Count which label appears most and assign it.
                label = np.argmax(np.bincount(data[:, -1].astype(int)))

            # Add vertex.
            self.add_vertices(
                vertices=[
                    {
                        'index': index,
                        'splitting_column_index': splitting_column_index,
                        'threshold': threshold,
                        'gini': best_gini,
                        'leaf': is_leaf,
                        'label': label
                    }
                ]
            )

            # Add edges for vertices that aren't leaves with indices given by
            # left_child_index = 2(parent_index) + 1
            # right_child_index = 2(parent_index) + 2.
            if not is_leaf:
                self.add_edges(
                    edges=[
                        {'parent_index': index, 'child_index': (2 * index) + 1},
                        {'parent_index': index, 'child_index': (2 * index) + 2}
                    ]
                )

            if not is_leaf:
                left_split, right_split = split(
                    data=data,
                    condition=data[:, splitting_column_index] < threshold
                )

                stack.append(((2 * index) + 1, left_split))
                stack.append(((2 * index) + 2, right_split))

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            tree_position = 0
            decision_made = False
            while not decision_made:
                vertex_location = [i for i in range(0, len(self.vertices)) if self.vertices[i]['index'] == tree_position][0]
                current_vertex = self.vertices[vertex_location]

                splitting_column_index = current_vertex['splitting_column_index']
                threshold = current_vertex['threshold']

                if current_vertex['leaf'] is True:
                    predictions.append(current_vertex['label'])
                    decision_made = True
                else:
                    if row[splitting_column_index] < threshold:
                        tree_position = (2 * tree_position) + 1
                    else:
                        tree_position = (2 * tree_position) + 2

        predictions = np.array(predictions).reshape(x_test.shape[0], -1)
        return np.array(predictions)


############
# Regression
############
class DecisionTreeRegression(Tree):
    """Decision tree regression"""
    def fit(self, x_train, y_train, tol=10.0):
        data = np.concatenate((x_train, y_train), axis=1)

        stack = [(0, data)]
        while len(stack) > 0:
            index, data = stack.pop(0)

            splitting_column_index, threshold, best_std_reduction = determine_best_split(data, 'r')

            is_leaf = False
            if coefficient_of_variation(data[:, -1]) <= tol:
                is_leaf = True

            # Assign values to leaves.
            value = None
            if is_leaf:
                # Assign the average of the target values in this split.
                value = np.mean(data[:, -1])

            # Add vertex.
            self.add_vertices(
                vertices=[
                    {
                        'index': index,
                        'splitting_column_index': splitting_column_index,
                        'threshold': threshold,
                        'std_reduction': best_std_reduction,
                        'leaf': is_leaf,
                        'value': value
                    }
                ]
            )

            # Add edges for vertices that aren't leaves with indices given by
            # left_child_index = 2(parent_index) + 1
            # right_child_index = 2(parent_index) + 2.
            if not is_leaf:
                self.add_edges(
                    edges=[
                        {'parent_index': index, 'child_index': (2 * index) + 1},
                        {'parent_index': index, 'child_index': (2 * index) + 2}
                    ]
                )

            if not is_leaf:
                left_split, right_split = split(
                    data=data,
                    condition=data[:, splitting_column_index] < threshold
                )

                stack.append(((2 * index) + 1, left_split))
                stack.append(((2 * index) + 2, right_split))

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            tree_position = 0
            decision_made = False
            while not decision_made:
                vertex_location = [i for i in range(0, len(self.vertices)) if self.vertices[i]['index'] == tree_position][0]
                current_vertex = self.vertices[vertex_location]

                splitting_column_index = current_vertex['splitting_column_index']
                threshold = current_vertex['threshold']

                if current_vertex['leaf'] is True:
                    predictions.append(current_vertex['value'])
                    decision_made = True
                else:
                    if row[splitting_column_index] < threshold:
                        tree_position = (2 * tree_position) + 1
                    else:
                        tree_position = (2 * tree_position) + 2

        predictions = np.array(predictions).reshape(x_test.shape[0], -1)
        return np.array(predictions)


# --- testing --- #
if __name__ == '__main__':
    data = [
        (3, 4, 0),
        (0.2, 2, 1),
        (2, 1.2, 0),
        (5, 2, 1)
    ]
    arr = np.array(data)
    x = arr[:, :-1]
    y = arr[:, -1].reshape(x.shape[0], -1)

    test = np.array([(4, 0.1), (5, 2.3)])

    model = DecisionTreeRegression()
    model.fit(x, y)
    print(model.predict(test))
