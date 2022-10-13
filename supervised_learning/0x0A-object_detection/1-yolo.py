#!/usr/bin/env python3
"""
0. Initialize Yolo
"""
import numpy as np
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

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs: np.ndarray contains predictions from the darknet
                model for a single image
                shape: (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width => height and width of grid
                        used for output
                    anchor_boxes => number of anchor boxes used
                    4 => (t_x, t_y, t_w, t_h)
                    1 => box_confidence
                    classes => class probabilities for all classes

            image_size: np.ndarray contains image's original size hxw

        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        def sigmoid(array):
            """ Sigmoid activation function """
            return 1 / (1 + np.exp(-1 * array))

        boxes, box_confidences, box_class_probs = [], [], []
        image_width = self.model.input.shape[1]
        image_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            output_boxes = output[..., :4]
            grid_height, grid_width, anchors = output.shape[:3]

            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            cx = np.repeat(cx[..., np.newaxis], anchors, axis=2)
            cy = np.arange(grid_width).reshape(1, grid_width)
            cy = np.repeat(cy, grid_height, axis=0)
            cy = np.repeat(cy[..., np.newaxis], anchors, axis=2)

            tx = output_boxes[..., 0]
            ty = output_boxes[..., 1]
            tw = output_boxes[..., 2]
            th = output_boxes[..., 3]

            ph = self.anchors[i, :, 0]
            pw = self.anchors[i, :, 1]

            bx = (sigmoid(tx) + cx) / grid_width
            by = (sigmoid(ty) + cy) / grid_height
            bw = (pw * np.exp(tw)) / image_width
            bh = (ph * np.exp(th)) / image_height

            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]

            output_boxes[..., 0] = x1
            output_boxes[..., 1] = y1
            output_boxes[..., 2] = x2
            output_boxes[..., 3] = y2

            boxes.append(output_boxes)
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append((sigmoid(output[..., 5:])))
        return boxes, box_confidences, box_class_probs
