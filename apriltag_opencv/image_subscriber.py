import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.declare_parameter('topic', 'image_raw')
        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.bridge = CvBridge()
        self.create_subscription(Image, self.topic, self.listener_callback, 10)

    def listener_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv.imshow("AprilTag Detection", cv_img)
        cv.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    cv.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
