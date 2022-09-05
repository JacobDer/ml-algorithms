import sys
import numpy as np
import math
from collections import Counter


def euclidean_distance(x, y):
    """Compute euclidean distance between two points."""
    # Check that x and y have the same dimension.
    if x.shape[0] != y.shape[0]:
        print('Error: Points must have the same dimension.')
        sys.exit()

    summands = []
    for i in range(0, x.shape[0]):
        summand = (x[i] - y[i])**2
        summands.append(summand)

    distance = math.sqrt(sum(summands))
    return distance


class KNNClassifier():
    """K nearest neighbors classifier."""
    def __init__(self, k):
        self.k = k
        self.observations = []
        self.labels = []

    def fit(self, x_train, y_train):
        # Training is done by storing observations and their labels.
        for row in range(0, x_train.shape[0]):
            self.observations.append(x_train[row])
            self.labels.append(y_train[row])

    def predict(self, x_test):
        predictions = []
        # Compute distance of each row in test data to each training data
        # point.
        for i in range(0, x_test.shape[0]):
            distance_rankings = []
            row = x_test[i]
            for j in range(0, len(self.observations)):
                observation = self.observations[j]
                label = self.labels[j]
                distance = euclidean_distance(row, observation)

                distance_ranking = (observation, label[0], distance, row)
                distance_rankings.append(distance_ranking)

            # Sort by distance and keep only the first k. This returns the k
            # nearest neighbors to the test point.
            distance_rankings = sorted(distance_rankings, key=lambda t: t[2])
            knn = distance_rankings[0: self.k]
            knn_labels = [tup[1] for tup in knn]

            # Retrieve the most common label among the knn.
            most_common_label = Counter(knn_labels).most_common(1)[0]
            predictions.append(most_common_label[0])

        predictions = np.array(predictions).reshape(x_test.shape[1], -1)
        return predictions


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

    test = np.array([(4, 1.4), (5, 0)])

    model = KNNClassifier(4)
    model.fit(x, y)
    print(model.predict(test))
