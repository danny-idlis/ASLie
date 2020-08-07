import base64
from io import BytesIO
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tf INFO and WARNING prints.
import tensorflow as tf
import numpy as np
from PIL import Image


class HandDetector:
    def __init__(self, pb_file='frozen_inference_graph.pb'):
        with tf.gfile.FastGFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def, name="")
        self.sess = tf.Session(graph=g_in)

    def _run(self, image):
        width, height = image.size
        resize_ratio = 256 / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        (boxes, scores, classes, num) = self.sess.run(
            ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'],
            feed_dict={'image_tensor:0': [np.asarray(resized_image)]})
        return np.squeeze(boxes), np.squeeze(scores)

    @staticmethod
    def _apply_threshold(boxes, scores, threshold):
        result_boxes = []
        result_scores = []
        for i in range(scores.shape[0]):
            if scores[i] > threshold:
                result_boxes.append(boxes[i].tolist())
                result_scores.append(scores[i].tolist())
        return result_boxes, result_scores

    def get_boxes(self, image, threshold=0.25):
        boxes, scores = self._run(image)
        return self._apply_threshold(boxes, scores, threshold=threshold)
