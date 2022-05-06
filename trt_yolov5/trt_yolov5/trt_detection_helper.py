# ROS2 imports
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

from trt_yolov5.trt_model import trt_yolov5
from trt_yolov5.export import ONNX_to_TRT

import cv2
import time
import os

import tensorrt as trt
import numpy as np


def str2bool(s):
    return s.lower() in ('true', '1')

class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class TRTDetectionNode(Node):
    
    def __init__(self) -> None:
        super().__init__('trt_detection_node')
        # Create a subscriber to the Image topic
        self.declare_parameter('topic', "image")
        topic = self.get_parameter('topic').value
        self.subscription = self.create_subscription(
            Image, topic, self.listener_callback, 10)
        
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(
            Detection2DArray, 'trt_detection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(
            Image, 'trt_detection_image', 10)
        
        self.timer = Timer()
        
        # base dir
        self.declare_parameter('base_dir', "/workspace")
        self.base_dir = self.get_parameter('base_dir').value
        
        # create a model parameter, by default the model is yolov5n
        self.declare_parameter('model', "yolov5n")
        self.model_name = self.get_parameter('model').value
        
        # iou threshold
        self.declare_parameter('iou', 0.4)
        self.iou_thresh = float(self.get_parameter('iou').value)
        # confidence threshold
        self.declare_parameter('conf', 0.2)
        self.conf_thresh = float(self.get_parameter('conf').value)
        # max memory
        self.declare_parameter('msize', 4)
        msize = int((self.get_parameter('msize').value ))
        
        print('Loading model')
        self.model = self.build_engine(
            model_name=self.model_name, fp16=False, size= msize)
        print('Start')
        # for rendering color
        self.colors = Colors()
        
        
        
    def build_engine(self, model_name, fp16= False, size=8):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        if trt.__version__[0] == '7':
            opset = 12
        else:  # TensorRT >= 8
            opset = 13
        onnx_path = os.path.join(self.base_dir, 'src/trt_yolov5/models', f'{model_name}_{opset}.onnx')
        trt_path = os.path.join(self.base_dir, 'src/trt_yolov5/models', f'{model_name}_{opset}.trt')
        try:
            model = trt_yolov5(model_path=trt_path, tensor_height=640, tensor_width=640)
            return model
        except:
            print('Engine not found -> build engine from Onnx')
            ONNX_to_TRT(onnx_path, trt_path, fp16, size)
            model = trt_yolov5(model_path=trt_path,
                               tensor_height=640, tensor_width=640)
            return model
            
    
    def listener_callback(self, data):
        self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.timer.start()
        label_ids, probs, boxes = self.model.predict(image, self.conf_thresh, self.iou_thresh, [])
        interval = self.timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, label_ids.shape[0]))

        detection_array = Detection2DArray()
        
        for i in range(len(boxes)):
            box = boxes[i]
            label = f"{self.model.classes[label_ids[i]]}: {probs[i]:.2f}"
            print("Object: " + str(i) + " " + label)
            color=self.colors(int(label_ids[i]) % len(self.model.classes))
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), color, 4)

            # Definition of 2D array message and ading all object stored in it.
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis_with_pose.id = str(self.model.classes[label_ids[i]])
            object_hypothesis_with_pose.score = float(probs[i])

            bounding_box = BoundingBox2D()
            bounding_box.center.x = float((box[0] + box[2])/2)
            bounding_box.center.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            
            bounding_box.size_x = float(2*(bounding_box.center.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.y - box[1]))

            detection = Detection2D()
            detection.header = data.header
            detection.results.append(object_hypothesis_with_pose)
            detection.bbox = bounding_box

            detection_array.header = data.header
            detection_array.detections.append(detection)


            cv2.putText(cv_image, label,
                       (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                       color, 2)  # line type
        # Displaying the predictions
        cv2.imshow('trt_yolov5', cv_image)
        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.detection_publisher.publish(detection_array)
        
        ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        ros_image.header.frame_id = 'camera_frame'
        self.result_publisher.publish(ros_image)
        cv2.waitKey(1)
