#!/usr/bin/env python3
"""
3. Non-max Suppression
"""
import cv2
import numpy as np
import os
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes
        Args:
            boxes: np.ndarray (grid_height, grid_width, anchor_boxes, 4)
                processed boundary boxes of each output
            box_confidences: np.ndarray (grid_height, grid_width,
                anchor_boxes, 1) - contains box confidences
            box_class_probs: np.ndarray (grid_height, grid_width,
                anchor_boxes, classes) - contains box class probabilities

        Returns: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for idx, box in enumerate(boxes):
            box_confidence = box_confidences[idx]
            box_class_prob = box_class_probs[idx]

            box_score = box_confidence * box_class_prob

            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            filter = np.where(box_class_score >= self.class_t)

            filtered_boxes.append(box[filter])
            box_classes.append(box_class[filter])
            box_scores.append(box_class_score[filter])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """ Calculates IOU """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

        # No overlap
        if w_intersection <= 0 or h_intersection <= 0:
            return 0

        intersection = w_intersection * h_intersection
        # Areas - I
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Args:
            filtered_boxes: np.ndarray (?, 4) - all filtered bounding boxes
            box_classes: np.ndarray (?,) - contains class number
            box_scores: np.ndarray (?) - contains box score

        Returns: (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        chosen = []
        indices = np.lexsort((box_scores, -box_classes))

        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            chosen.append(i)
            suppress = [last]

            for pos in range(last):
                j = indices[pos]
                if box_classes[i] == box_classes[j]:
                    if self.iou(filtered_boxes[i],
                                filtered_boxes[j]) > self.nms_t:
                        suppress.append(pos)
            indices = np.delete(indices, suppress)
        return filtered_boxes[chosen], box_classes[chosen], box_scores[chosen]

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from path
        Args:
            folder_path: string representing the path to the folder
                holding all the images to load

        Returns: (images, image_paths)
        """
        images, image_paths = [], []
        img_path = folder_path + "/"
        image_path = os.listdir(folder_path)

        for path in image_path:
            images.append(cv2.imread(img_path + path))
            image_paths.append(img_path + path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Processes images
        Args:
            images: list of images as np.ndarrays

        Returns: (pimages, image_shapes)
        """
        image_w = self.model.input.shape[1]
        image_h = self.model.input.shape[2]
        pimages, image_shapes = [], []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            pimages.append(cv2.resize(
                image,
                dsize=(image_w, image_h),
                interpolation=cv2.INTER_CUBIC
            ))
        pimages = [x / 255 for x in pimages]

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names,
            and box scores
        Args:
            image: np.ndarray - unprocessed image
            boxes: np.ndarray - boundary boxes
            box_classes: np.ndarray - class indices
            box_scores: np.ndarray - box scores
            file_name: path where original image is stored

        Returns:
        """
        for i in range(boxes.shape[0]):
            # Get the rectangle coords
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            coord1 = (x1, y1)
            coord2 = (x2, y2)
            blue_rect = (255, 0, 0)
            # Draw rectangle
            image = cv2.rectangle(image, coord1, coord2,
                                  color=blue_rect, thickness=2)
            # Text
            box_score = np.around(box_scores, 2)[i]
            class_name = self.class_names[box_classes[i]]
            text = str(class_name) + " " + str(box_score)
            # Text should be written 5 pixels above the
            # top left corner of the box
            text_coord = (x1, y1 - 5)
            red_font = (0, 0, 255)
            image = cv2.putText(image, text, text_coord,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, red_font, 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord("s"):
            try:
                os.mkdir("detections")
            except FileExistsError:
                pass
            cv2.imwrite("detections/" + file_name, image)
        cv2.destroyAllWindows()
