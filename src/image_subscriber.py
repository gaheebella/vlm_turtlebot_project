import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from clip_navigator import compute_direction_scores, decide_velocity
from clip_core import load_clip_model


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        loaded = load_clip_model()
        if len(loaded) == 4:
            self.model, self.preprocess, _, self.device = loaded
        elif len(loaded) == 3:
            self.model, self.preprocess, self.device = loaded
        else:
            raise ValueError("load_clip_model() 반환값 개수를 확인하세요.")

        self.goal_text = "a chair"
        self.frame_count = 0
        self.infer_interval = 5

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("ImageSubscriber with CLIP navigation started")

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.infer_interval != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return

        try:
            scores = compute_direction_scores(
                frame_bgr=frame,
                text_goal=self.goal_text,
                model=self.model,
                preprocess=self.preprocess,
                device=self.device,
            )

            linear_v, angular_v, best_name, margin, mode = decide_velocity(
                scores=scores,
                obstacle_penalty=0.0
            )

            twist = Twist()
            twist.linear.x = float(linear_v)
            twist.angular.z = float(angular_v)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"scores={scores}, best={best_name}, margin={margin:.4f}, "
                f"mode={mode}, v={linear_v:.2f}, w={angular_v:.2f}"
            )

            vis = frame.copy()
            text1 = f"Goal: {self.goal_text}"
            text2 = f"L:{scores['left']:.3f} C:{scores['center']:.3f} R:{scores['right']:.3f}"
            text3 = f"Best: {best_name}, Margin: {margin:.3f}, Mode: {mode}"
            text4 = f"cmd_vel -> v:{linear_v:.2f}, w:{angular_v:.2f}"

            cv2.putText(vis, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis, text4, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            h, w = vis.shape[:2]
            cv2.line(vis, (int(w * 0.4), 0), (int(w * 0.4), h), (255, 0, 0), 2)
            cv2.line(vis, (int(w * 0.3), 0), (int(w * 0.3), h), (100, 100, 255), 1)
            cv2.line(vis, (int(w * 0.6), 0), (int(w * 0.6), h), (255, 0, 0), 2)
            cv2.line(vis, (int(w * 0.7), 0), (int(w * 0.7), h), (100, 100, 255), 1)

            cv2.imshow("CLIP Navigator", vis)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"inference/control 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()