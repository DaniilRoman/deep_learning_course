import pandas as pd
from knn import OwnKNeighborsClassifier, euclidian_distance

def best_score():
    train = pd.read_csv("./data/archive/mnist_train.csv")
    test = pd.read_csv("./data/archive/mnist_test.csv")

    X_train = train.drop(['label'], axis=1).values
    y_train = train['label'].values

    X_test = test.drop(['label'], axis=1).values
    y_test = test['label'].values

    y_train = y_train.reshape(train.shape[0], 1)
    y_test = y_test.reshape(test.shape[0], 1)

    from hog import HogFeatureLoader

    hog = HogFeatureLoader()
    hog.save_features(X_train, "full_train")
    hog.save_features(X_test, "full_test")

    x_train = hog.load_features("full_train")
    x_test = hog.load_features("full_test")

    model = OwnKNeighborsClassifier(3, euclidian_distance)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"Accuracy: {score}")