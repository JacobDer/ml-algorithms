import numpy as np
from random import randint
from collections import Counter
from decision_tree import DecisionTreeClassifier, DecisionTreeRegression


def sample_data(x_train, y_train, sample_size):
    """Sample data with replacement."""
    indices = [randint(0, x_train.shape[0] - 1) for _ in range(0, sample_size)]

    x_train_sample = x_train[indices]
    y_train_sample = y_train[indices]
    return (x_train_sample, y_train_sample)


class RandomForestClassifier():
    """Random forest classifier"""
    def __init__(self):
        self.decision_trees = []
        self.votes = []

    def fit(self, x_train, y_train, no_trees, sample_size, tol=0.0):
        # Create decision trees and train them on sampled data.
        for _ in range(0, no_trees):
            x_train_sample, y_train_sample = sample_data(
                x_train=x_train,
                y_train=y_train,
                sample_size=sample_size
            )

            decision_tree = DecisionTreeClassifier()
            decision_tree.fit(
                x_train=x_train_sample,
                y_train=y_train_sample,
                tol=tol
            )

            self.decision_trees.append(decision_tree)

    def predict(self, x_test):
        # Each decision submits a prediction vote.
        for decision_tree in self.decision_trees:
            vote = decision_tree.predict(x_test)
            self.votes.append(vote)

        # Count the most common prediction vote for each input.
        most_common_votes = []
        for row in range(0, x_test.shape[0]):
            row_votes = []
            for vote in self.votes:
                row_vote = vote[row][0]
                row_votes.append(row_vote)

            most_common_row_vote = Counter(row_votes).most_common(1)[0]
            most_common_votes.append(most_common_row_vote[0])

        prediction = np.array(most_common_votes).reshape(x_test.shape[0], -1)
        return prediction


class RandomForestRegression():
    """Random forest classifier"""
    def __init__(self):
        self.decision_trees = []
        self.votes = []

    def fit(self, x_train, y_train, no_trees, sample_size, tol=10.0):
        # Create decision trees and train them on sampled data.
        for _ in range(0, no_trees):
            x_train_sample, y_train_sample = sample_data(
                x_train=x_train,
                y_train=y_train,
                sample_size=sample_size
            )

            decision_tree = DecisionTreeRegression()
            decision_tree.fit(
                x_train=x_train_sample,
                y_train=y_train_sample,
                tol=tol
            )

            self.decision_trees.append(decision_tree)

    def predict(self, x_test):
        # Each decision submits a prediction vote.
        for decision_tree in self.decision_trees:
            vote = decision_tree.predict(x_test)
            self.votes.append(vote)

        # Average the prediction votes for each input.
        average_votes = []
        for row in range(0, x_test.shape[0]):
            row_votes = []
            for vote in self.votes:
                row_vote = vote[row][0]
                row_votes.append(row_vote)

            average_row_vote = sum(row_votes) / len(row_votes)
            average_votes.append(average_row_vote)

        prediction = np.array(average_votes).reshape(x_test.shape[0], -1)
        return prediction


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

    test = np.array([(4, 1.4), (5, 0.2)])

    model = RandomForestRegression()
    model.fit(x, y, no_trees=64, sample_size=3, tol=10.0)
    print(model.predict(test))
