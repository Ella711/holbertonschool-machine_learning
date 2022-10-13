#!/usr/bin/env python3
"""
0. Initialize Yolo
"""
import tensorflow.keras as K


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names is stored
            class_t: float representing the box score threshold for the
                initial filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors - np.ndarray - shape (outputs, anchor_boxes, 2):
                contains all anchor boxes
                outputs: number of predictions made by the Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2 => [anchor_box_width, anchor_box_height]
        """
        class_names = []
        model = K.models.load_model(model_path)

        with open(classes_path, 'r', encoding='utf') as file:
            for line in file:
                class_names.append(line.strip())

        self.model = model
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
