from skimage.feature import hog
from skimage import exposure
import numpy as np

class HogFeatureLoader:

    def __init__(self, is_rescale=False):
        self.PIXELS_PER_CELL = 4
        self.ORIENTATIONS = 4
        self.CELLS_PER_BLOCKS = 1
        self.is_rescale = is_rescale

    def __get_feature(self, image):
        fd, hog_image = hog(image, orientations=self.ORIENTATIONS, pixels_per_cell=(self.PIXELS_PER_CELL, self.PIXELS_PER_CELL),
                            cells_per_block=(self.CELLS_PER_BLOCKS, self.CELLS_PER_BLOCKS), visualize=True, multichannel=False)
        if self.is_rescale:
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return hog_image


    def __get_features(self, images):
        result = []

        for img in images:
            len_ = len(img)
            new_len_ = int(np.sqrt(len_))

            img = img.reshape(new_len_, new_len_)

            img = self.__get_feature(img)
            result.append(img.reshape(len_))

        return np.array(result)


    def save_features(self, images, prefix):
        features = self.__get_features(images)
        np.savetxt(f"data/features/hog_{prefix}.csv", features, delimiter=",")


    def load_features(self, prefix):
        return np.genfromtxt(f"data/features/hog_{prefix}.csv", delimiter=',')

