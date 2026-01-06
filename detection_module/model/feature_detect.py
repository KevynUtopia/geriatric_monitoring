import numpy as np
import model.landmark_indices as landmark_indices


class HeuristicFeatureDetector:
    def __init__(self, eye_threshold, mouth_threshold):
        self.eye_threshold = eye_threshold
        self.mouth_threshold = mouth_threshold

    def detectEyeOpen(self, face_landmarks):
        left_EAR = self._calcAspectRatio(
            face_landmarks[landmark_indices.LEFT_EYE])
        right_EAR = self._calcAspectRatio(
            face_landmarks[landmark_indices.RIGHT_EYE])

        return np.mean([left_EAR, right_EAR]) >= self.eye_threshold

    def detectMouthOpen(self, face_landmarks):
        mouth_AR = self._calcAspectRatio(
            face_landmarks[landmark_indices.MOUTH])

        return mouth_AR >= self.mouth_threshold, mouth_AR

    def _calcAspectRatio(self, feature_landmarks):
        hori = np.linalg.norm(feature_landmarks[0] -
                              feature_landmarks[len(feature_landmarks) // 2])
        vert = [
            np.linalg.norm(feature_landmarks[i] - feature_landmarks[-i])
            for i in range(1,
                           len(feature_landmarks) // 2)
        ]

        return sum(vert) / (2 * hori)
