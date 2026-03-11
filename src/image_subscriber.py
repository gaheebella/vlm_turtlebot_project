import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        print("ImageSubscriber started")

        self.bridge = CvBridge()

        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos
        )

        self.count = 0
        os.makedirs('dataset/raw_frames', exist_ok=True)

    def image_callback(self, msg):
        print("callback received")

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("frame shape:", frame.shape)

        cv2.imshow('Webcam Image', frame)
        cv2.waitKey(1)

        if self.count % 30 == 0:
            filename = f'dataset/raw_frames/frame_{self.count:04d}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Saved: {filename}')

        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()