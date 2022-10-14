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

    @staticmethod
    def sigmoid(array):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-1 * array))

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
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            # Finding the boxes
            output_boxes = output[..., :4]

            # Get grid width and height : number of grid cells
            g_w, g_h = output.shape[:2]

            # Anchor of the current output
            anchors = self.anchors[i]

            txy = output_boxes[..., :2]
            twh = output_boxes[..., 2:4]

            # Grid cell indices
            grid = np.tile(np.indices((g_w, g_h)).T, 3)
            grid = grid.reshape((g_h, g_w) + anchors.shape)

            # Finding center of each bounding box per cell
            bxy = self.sigmoid(txy) + grid
            bwh = anchors * np.exp(twh)

            # Normalize bxy and bwh
            # bwh: divide by model's input shape
            bwh /= self.model.inputs[0].shape.as_list()[1:3]
            # bxy: divide by the grid size
            bxy /= [g_w, g_h]

            # Find corners
            # Top left
            bxy1 = bxy - (bwh / 2)
            # Bottom right
            bxy2 = bxy + (bwh / 2)

            box = np.concatenate((bxy1, bxy2), axis=-1)

            # Multiply by original image size
            box = box * np.tile(np.flip(image_size, axis=0), 2)
            boxes.append(box)

            confidence = np.expand_dims(self.sigmoid(output[..., 4]), axis=-1)
            box_confidences.append(confidence)

            box_class_probs.append((self.sigmoid(output[..., 5:])))
        return boxes, box_confidences, box_class_probs
