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

        for row in (output_data):
            #row = output_data[r]
            #print(row)
            
            confidence = row[4]
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if confidence >= conf_thresh \
                and (class_id in accepted_classes or accepted_classes == []):
                    
                if (classes_scores[class_id] >= conf_thresh):
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height], dtype=int)
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        for i in indexes:
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
