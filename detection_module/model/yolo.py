import os
import numpy as np
import cv2
from absl import logging
from model.yolo_model import eval
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from PIL import Image


class YOLO(object):
    def __init__(self, model_path, classes_path, anchors_path, iou, score):
        self.model_path = model_path
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(self.sess)
        self.boxes, self.scores, self.classes = self._generate(iou, score)

    def detectImage(self, image, image_size=None):
        image = Image.fromarray(image)
        if image_size is None:
            new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
        else:
            new_image_size = (image_size[0] - (image_size[0] % 32),
                            image_size[1] - (image_size[1] % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch dimension
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        out_boxes = out_boxes[:, [1, 0, 3, 2]]
        return np.array(out_boxes, dtype=int)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self, iou, score):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file'

        # load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (
                           num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        logging.info(
            '*** {} model, anchors, and classes loaded.'.format(model_path))

        # generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = eval(self.yolo_model.output,
                                      self.anchors,
                                      len(self.class_names),
                                      self.input_image_shape,
                                      score_threshold=score,
                                      iou_threshold=iou)
        return boxes, scores, classes

    def close_session(self):
        self.sess.close()


def letterbox_image(image, size):
    '''Resize image with unchanged aspect ratio using padding'''

    img_width, img_height = image.size
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
