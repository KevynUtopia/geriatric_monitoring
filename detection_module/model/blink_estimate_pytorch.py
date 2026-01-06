import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import model.landmark_indices as landmark_indices
from model.rt_gene import BlinkEstimatorPytorch


class BlinkEstimatorImageLandmarkPytorch(BlinkEstimatorPytorch):
    def __init__(
        self,
        threshold,
        input_size,
        crop_offset,
        model_path):
        super(BlinkEstimatorImageLandmarkPytorch,
              self).__init__("cuda:0", model_path, 'resnet18', threshold)
        self.threshold = threshold
        self.crop_offset = crop_offset
        self.input_size = input_size

    def resize_img(self, img):
        return cv2.resize(img,
                          dsize=self.input_size,
                          interpolation=cv2.INTER_CUBIC)

    def cropping(self, img, rect):
        # crop the image with a given rectangle
        x, y, w, h = rect
        if x < 0 or y < 0:
            raise ValueError("Invalid rectangle")

        crop = img[y - self.crop_offset : y+h + self.crop_offset,
               x - self.crop_offset : x+w + self.crop_offset]
        return crop

    def estimate(self, images, landmarks):
        landmarks = np.array(landmarks, dtype=np.int32)
        # crop = lambda img, rect: img[rect[1] - self.crop_offset:rect[1] + rect[
        #     3] + self.crop_offset, rect[0] - self.crop_offset:rect[0] + rect[2]
        #                              + self.crop_offset]
        left_images = []
        right_images = []

        for image, landmark in zip(images, landmarks):
            left_rect = cv2.boundingRect(landmark[landmark_indices.LEFT_EYE])
            right_rect = cv2.boundingRect(landmark[landmark_indices.RIGHT_EYE])

            left, right = self.inputs_from_images(self.cropping(image, left_rect), self.cropping(image, right_rect))
            left_images.append(left)
            right_images.append(right)

        probs = self.predict(left_images, right_images)
        # blinks = p >= self.threshold
        blinks = None
        return probs, blinks
