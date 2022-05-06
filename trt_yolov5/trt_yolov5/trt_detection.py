
import rclpy
from trt_yolov5.trt_detection_helper import TRTDetectionNode


def main(args=None):
    rclpy.init(args=args)

    trt_detection_node = TRTDetectionNode()

    rclpy.spin(trt_detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trt_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
