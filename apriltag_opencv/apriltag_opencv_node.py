#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pupil_apriltags import Detector
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import numpy as np

class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')
        
        # 必要な初期化
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'image_raw', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # タグの位置とクオータニオンを格納する行列
        self.tag_xyz = np.zeros((10, 3))  # 最大10個のタグに対応
        self.tag_euler = np.zeros((10, 3))  # オイラー角用

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('Failed to capture image')
            return
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(
            gray_frame,
            estimate_tag_pose=True,
            camera_params=(640, 480, 320, 240),
            tag_size=0.162
        )
        
        for i, tag in enumerate(tags):
            # タグのID
            tag_id = tag.tag_id

            # タグの位置と姿勢を行列に保存
            self.tag_xyz[i][0] = tag.pose_t[0]
            self.tag_xyz[i][1] = tag.pose_t[1]
            self.tag_xyz[i][2] = tag.pose_t[2]

             # 回転行列をオイラー角に変換して行列に保存
            rotation_matrix = np.array(tag.pose_R)
            euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
            self.tag_euler[i] = euler_angles

            # タグの位置とオイラー角を表示
            self.get_logger().info(f"Tag {tag_id} Position: [{self.tag_xyz[i][0]:.3f}, {self.tag_xyz[i][1]:.3f}, {self.tag_xyz[i][2]:.3f}]")
            self.get_logger().info(f"Tag {tag_id} Euler Angles: [{self.tag_euler[i][0]:.3f}, {self.tag_euler[i][1]:.3f}, {self.tag_euler[i][2]:.3f}]")

            # 描画
            center = (int(tag.center[0]), int(tag.center[1]))
            corners = [(int(c[0]), int(c[1])) for c in tag.corners]
            cv2.circle(frame, center, 5, (0, 0, 255), 2)
            for j in range(4):
                cv2.line(frame, corners[j], corners[(j + 1) % 4], (0, 255, 0), 2)
            cv2.putText(frame, str(tag_id), (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            # Transformの作成とブロードキャスト
            transform = geometry_msgs.msg.TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'camera_frame'
            transform.child_frame_id = f'tag_{tag_id}'

            transform.transform.translation.x = self.tag_xyz[i][0]
            transform.transform.translation.y = self.tag_xyz[i][1]
            transform.transform.translation.z = self.tag_xyz[i][2]
            
            # クオータニオンから取得した回転をブロードキャスト
            quaternion = R.from_euler('xyz', self.tag_euler[i], degrees=True).as_quat()
            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]

            self.tf_broadcaster.sendTransform(transform)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetector()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
