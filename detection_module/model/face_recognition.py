import os
from absl import logging
import numpy as np
import cv2
import face_recognition
import dlib
import model.facial_process as facial_process


class FaceRecognizer:
    def __init__(self, face_dataset="data/faces/", frontalize=False, dpath="weights/mmod_human_face_detector.dat", epath="weights/dlib_face_recognition_resnet_model_v1.dat"):
        detector = dlib.cnn_face_detection_model_v1(dpath)
        self.encoder = dlib.face_recognition_model_v1(epath)
        self.face_dataset_encoding = []
        self.datasetidx_to_person = []

        face_paths = os.listdir(face_dataset)
        logging.info("Face recognizer ~ Found {} faces in dataset".format(
            len(face_paths)))
        if frontalize:
            logging.info("Face recognizer ~ Frontalizing face in creating dataset")


        for path in sorted(face_paths):
            # index of the this face
            # multi faces could indicate the same person
            person = path.split(".")[0]
            if not person.isdigit():
                person = person[:-1]

            image = cv2.imread(face_dataset + path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dlib_box = detector(image, 1)[0]
            if dlib_box is None:
                continue
            box = [dlib_box.rect.left(), dlib_box.rect.top(), dlib_box.rect.right(), dlib_box.rect.bottom()]
            landmark = facial_process.processLandmark(image, boxes=[box])[0]

            if frontalize:
                image, _, _ = facial_process.frontalizeFace(image, landmarks)
                dlib_box = detector(image, 1)[0]
                if dlib_box is None:
                    continue
                box = [dlib_box.rect.left(), dlib_box.rect.top(), dlib_box.rect.right(), dlib_box.rect.bottom()]
                landmark = facial_process.processLandmark(image, boxes=[box])[0]

            try:
                encoding = self._getEncoding(image, dlib_box.rect, landmark)
            except IndexError:
                logging.error("Face recognizer ~ Error processing {}".format(face_dataset + path))
                continue

            self.datasetidx_to_person.append(person)
            self.face_dataset_encoding.append(encoding)

        self.face_dataset_encoding = np.array(self.face_dataset_encoding)
        del detector

    def _getEncoding(self, image, rect, landmark):
        if type(rect) != dlib.rectangle:
            dlib_face_rect = dlib.rectangle(*rect)
        else:
            dlib_face_rect = rect
        landmark_dlib = dlib.full_object_detection(rect=dlib_face_rect, parts=[dlib.point(point) for point in landmark])
        encoding = np.array(self.encoder.compute_face_descriptor(image, landmark_dlib, 1))
        
        return encoding

    def recognizeFaceBatch(self, image, face_boxes, landmarks, tolerance=0.8):
        face_boxes = face_boxes[:, [1, 2, 3, 0]]
        
        encodings = [self._getEncoding(image, box, lm) for box, lm in zip(face_boxes, landmarks)]

        euc_distances = np.array([np.linalg.norm(self.face_dataset_encoding - encoding, axis=1) for encoding in encodings])

        min_dists, min_dist_ids = np.min(euc_distances, axis=1), np.argmin(euc_distances, axis=1)
        identified_people = [self.datasetidx_to_person[idx] for (idx) in min_dist_ids]

        # If two people identified in the same frame, use the one with smallest distance
        logging.debug("Initial {}".format(identified_people))
        person_smallestidx = {}
        for i, (person, dist) in enumerate(zip(identified_people, min_dists)):
            if person not in person_smallestidx:
                person_smallestidx[person] = i
                continue

            if dist < min_dists[person_smallestidx[person]]:
                larger_idx = person_smallestidx[person]
                person_smallestidx[person] = i
            else:
                larger_idx = i
            identified_people[larger_idx] = None

        return identified_people, min_dists