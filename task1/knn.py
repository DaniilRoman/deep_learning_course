import numpy as np
from tqdm import tqdm
from collections import Counter



def euclidian_distance(a, b):
    return np.linalg.norm(a-b)

def manhattan_distance(a, b):
    return np.sum(np.abs(a-b))



class OwnKNeighborsClassifier:

    def __init__(self, k=1, distance_func=euclidian_distance):
        if k > 0:
            self.k = k
        else:
            raise ValueError('Please provide a valid value for k. K must be more then zero.')
        self.distance_func = distance_func

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "Equal number of samples and labels expected"

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):

        labels = []

        for x in tqdm(X):
            dist = [self.distance_func(x, x_train) for x_train in self.X_train]

            k_samples = np.argsort(dist)[:self.k]

            k_labels = [self.y_train[i][0] for i in k_samples]

            label = Counter(k_labels).most_common(1)[0][0]

            labels.append(label)

        return labels

    def predict_proba(self, X):
        labels_proba = []

        for x in tqdm(X):
            dist = [self.distance_func(x, x_train) for x_train in self.X_train]

            k_samples = np.argsort(dist)[:self.k]

            k_labels = [self.y_train[i][0] for i in k_samples]

            proba = np.zeros(10)
            label_proba = Counter(k_labels).items()
            for p in label_proba:
                proba[p[0]] = p[1]/self.k


            labels_proba.append(proba.tolist())

        return labels_proba

    def score(self, X, y):
        assert X.shape[0] == y.shape[0], "Equal number of samples and labels expected"

        labels = self.predict(X)

        total = 0
        correct = 0

        for i in range(len(labels)):
            total += 1

            if labels[i] == y[i]:
                correct += 1

        return correct / total

