import numpy as np


class HistFeatureLoader:

    def __init__(self, is_range=False):
        self.BINS = 56
        self.LEFT_RANGE_LIMIT = 0
        self.RIGHT_RANGE_LIMIT = 243
        self.is_range = is_range


    def __get_hist_feature(self, image):
        if self.is_range:
            hist = np.histogram(image, self.BINS, range=(self.LEFT_RANGE_LIMIT, self.RIGHT_RANGE_LIMIT))
        else:
            hist = np.histogram(image, self.BINS)
        return hist[1]


    def __get_hist_features(self, images):
        result = []

        for img in images:
            img = self.__get_hist_feature(img)
            result.append(img)

        return np.array(result)

    def save_features(self, images, prefix):
        features = self.__get_hist_features(images)
        np.savetxt(f"data/features/hist_{prefix}.csv", features, delimiter=",")


    def load_features(self, prefix):
        return np.genfromtxt(f"data/features/hist_{prefix}.csv", delimiter=',')


