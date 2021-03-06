import numpy as np
import cv2

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class yolov5_base:
    coco_classes = COCO_CLASSES

    def building_model(self):
        raise NotImplemented

    def detect(self):
        raise NotImplemented

    @staticmethod
    def wrap_detection(input_image, output_data,
                       input_tensor_height=640, input_tensor_width=640,
                       conf_thresh=0.25, iou_thresh=0.45,
                       accepted_classes: list = [0]
                       ) -> tuple:
        """detect

        Args:
            input_image (np.ndarray): image
            output_data (np.ndarray): tensor
            input_tensor_height (int, optional): input tensor heigh. Defaults to 640.
            input_tensor_width (int, optional): input tensor widht. Defaults to 640.
            conf_thresh (float, optional): confidence threshold. Defaults to 0.25.
            iou_thresh (float, optional): iou threshold. Defaults to 0.45.
            accepted_classes (list, optional): index classes to detect. Defaults to [0].

        Returns:
            list: ids, confs, boxes
        """
        class_ids = []
        confidences = []
        boxes = []

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / input_tensor_width
        y_factor = image_height / input_tensor_height

        confidences = output_data[:, 4]
        accepted_conf_idx = confidences > conf_thresh
        x, y, w, h = output_data[accepted_conf_idx, 0], output_data[accepted_conf_idx,
                                                                    1], output_data[accepted_conf_idx, 2], output_data[accepted_conf_idx, 3]
        left = ((x - 0.5 * w) * x_factor).reshape(1, -1)
        top = ((y - 0.5 * h) * y_factor).reshape(1, -1)
        right = (left + (w * x_factor)).reshape(1, -1)
        bottom = (top + (h * y_factor)).reshape(1, -1)
        class_ids = np.argmax(output_data[accepted_conf_idx, 5:], axis=1)
        confidences = confidences[accepted_conf_idx]
        boxes = np.concatenate([left, top, right, bottom], axis=0).astype('int').T
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        for i in indexes:
            if isinstance(i, np.ndarray):
                i = i[0]
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    @staticmethod
    def format_yolov5(frame: np.ndarray) -> np.ndarray:
        """ Convert to rectangle frame (H & W are different) -> square frame (H=W)

        Args:
            frame (np.ndarray): numpy image

        Returns:
            np.ndarray: numpy image
        """
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    @staticmethod
    def load_classes(txt_classes_path: str) -> list:
        class_list = []
        with open(txt_classes_path, "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list
