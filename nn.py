from keras.applications.vgg16 import VGG16
import numpy as np


class NnFeatureLoader:

    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    def describe_model(self):
        self.model.summary()

    def __get_feature(self, image):
        image = np.resize(image, (1, 224, 224, 3))
        return self.model.predict(image).flatten()

    def __get_features(self, images):
        images = np.dstack([images] * 3)
        images = images.reshape(-1, 28, 28, 3)
        images = images.astype('float32')
        result = []

        for img in images:
            img = self.__get_feature(img)
            result.append(img)

        return np.array(result)

    def save_features(self, images, prefix):
        features = self.__get_features(images)
        np.savetxt(f"data/features/nn_{prefix}.csv", features, delimiter=",")

    def load_features(self, prefix):
        return np.genfromtxt(f"data/features/nn_{prefix}.csv", delimiter=',')
