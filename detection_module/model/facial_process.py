import os
from absl import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as io
import face_alignment
from model.face_frontalization import frontalize, camera_calibration

cur_path = os.path.dirname(os.path.abspath(__file__))
model = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType.TWO_D,
                                     flip_input=False,
                                     face_detector="dlib",
                                     device="cuda:0")
model3D = frontalize.ThreeD_Model(
    cur_path + "/face_frontalization/frontalization_models/model3Ddlib.mat",
    'model_dlib')
eyemask = np.asarray(
    io.loadmat(cur_path +
               "/face_frontalization/frontalization_models/eyemask.mat")
    ['eyemask'])


def processLandmark(image, boxes=None):
    landmarks = model.get_landmarks_from_image(image, detected_faces=boxes)
    # landmarks is a set of points
    # plots these points on the image with their index
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # for i, mark in enumerate(landmarks[0]):
    #     plt.text(mark[0], mark[1], str(i), fontsize=7, color='red')
    # plt.show()
    # # save with high resolution
    # plt.savefig("landmark.jpg", dpi=300)
    return landmarks


def frontalizeFace(image, landmarks):
    proj_matrix, camera_matrix, rmat, tvec = camera_calibration.estimate_camera(
        model3D, landmarks)
    frontal_raw, frontal_sym = frontalize.frontalize(image, proj_matrix,
                                                     model3D.ref_U, eyemask)

    return frontal_raw, frontal_sym, proj_matrix


def cropFace(image, box, landmark=None, offset=0):
    height, width = image.shape[:-1]
    x1, y1, x2, y2 = box
    x1 = max(int(x1 - offset * (x2 - x1)), 0)
    y1 = max(int(y1 - offset * (y2 - y1)), 0)
    x2 = min(int(x2 + offset * (x2 - x1)), width)
    y2 = min(int(y2 + offset * (y2 - y1)), height)
    face_image = image[y1:y2, x1:x2]

    return face_image, (landmark - [x1, y1]) if landmark is not None else None


def drawMarks(image, marks):
    plt.imshow(image)
    plt.axis("off")
    plt.plot(marks[:, 0], marks[:, 1], 'wo')
