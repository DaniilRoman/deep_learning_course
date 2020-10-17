import random
from task1.knn import OwnKNeighborsClassifier


class GetModelResult:
    def get_result(self,  k, feature, distance):
        model = OwnKNeighborsClassifier(k, distance)
        model.fit(feature["x_train"], feature["y_train"])
        return model.score(feature["x_test"], feature["y_test"])


class GridSearch(GetModelResult):

    def find(self, distance_functions, features, K):
        result_list = []
        for distance in distance_functions:
            for feature in features:
                for k in K:
                    result = self.get_result(k, feature[1], distance[1])
                    result_list.append([result, [distance[0], feature[0], k]])
        return {k: v for k, v in sorted(result_list, key=lambda item: item[0], reverse=True)}


class RandomSearch(GetModelResult):

    def find(self, distance_functions, features, K):
        result_list = []

        for i in range(10):
            distance = random.choice(distance_functions)
            feature = random.choice(features)
            k = random.choice(K)

            result = self.get_result(k, feature[1], distance[1])
            result_list.append([result, [distance[0], feature[0], k]])
        return {k: v for k, v in sorted(result_list, key=lambda item: item[0], reverse=True)}

