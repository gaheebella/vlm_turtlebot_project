import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

from clip_navigator import ClipDirectionNavigator
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

        self.navigator = ClipDirectionNavigator(
            model=self.model,
            preprocess=self.preprocess,
            device=self.device,
            model_name="ViT-B-32",
            history_size=5,
            center_bias=0.015,
            detect_threshold=0.24,
            margin_threshold=0.03,
            stop_area_ratio=0.38,
        )

        self.goal_text = "a chair"
        self.goal_lock = threading.Lock()

        self.nav_state = "SEARCHING"
        self.search_turn_dir = 1
        self.lost_count = 0
        self.detect_count = 0

        self.frame_count = 0
        self.infer_interval = 1

        self.latest_penalty = 0.0

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

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.input_thread = threading.Thread(
            target=self._goal_input_loop,
            daemon=True
        )
        self.input_thread.start()

        self.get_logger().info(f"CLIP Navigator started | goal: '{self.goal_text}'")
        self.get_logger().info("상태 기반 탐색/정렬/접근 모드로 실행됩니다.")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        total = len(ranges)

        if total == 0:
            self.latest_penalty = 0.0
            return

        front_count = max(1, int(total * 20 / 360))
        front = np.concatenate([ranges[-front_count:], ranges[:front_count]])
        front = front[np.isfinite(front)]

        if len(front) == 0:
            self.latest_penalty = 0.0
            return

        min_dist = float(np.min(front))
        threshold = 0.5

        if min_dist < threshold:
            self.latest_penalty = (threshold - min_dist) / threshold
        else:
            self.latest_penalty = 0.0

    def _goal_input_loop(self):
        while True:
            try:
                with self.goal_lock:
                    current_goal = self.goal_text

                new_goal = input(f"\n새 목표 입력 (현재: '{current_goal}'): ").strip()
                if new_goal:
                    with self.goal_lock:
                        self.goal_text = new_goal

                    self.nav_state = "SEARCHING"
                    self.lost_count = 0
                    self.detect_count = 0

                    self.get_logger().info(f"목표 변경 → '{new_goal}'")
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"goal input 오류: {e}")
                break

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.infer_interval != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return

        with self.goal_lock:
            current_goal = self.goal_text

        try:
            scores, meta = self.navigator.compute_direction_scores(
                frame_bgr=frame,
                text_goal=current_goal
            )

            if meta["target_visible"]:
                self.detect_count += 1
                self.lost_count = 0
            else:
                self.lost_count += 1
                self.detect_count = 0

            if self.lost_count > 25:
                self.search_turn_dir *= -1
                self.lost_count = 0
                self.get_logger().warn("목표를 오래 못 찾아 탐색 회전 방향을 반전합니다.")

            linear_v, angular_v, next_state, mode = self.navigator.decide_stateful_velocity(
                nav_state=self.nav_state,
                scores=scores,
                meta=meta,
                obstacle_penalty=self.latest_penalty,
                search_turn_dir=self.search_turn_dir,
            )

            self.nav_state = next_state

            twist = Twist()
            twist.linear.x = float(linear_v)
            twist.angular.z = float(angular_v)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"[STATE={self.nav_state}] goal='{current_goal}' | "
                f"L:{scores['left']:.3f} C:{scores['center']:.3f} R:{scores['right']:.3f} | "
                f"raw={meta['best_raw']} smooth={meta['best_smooth']} "
                f"score={meta['best_score']:.3f} margin={meta['margin']:.4f} "
                f"visible={meta['target_visible']} area={meta['approx_area_ratio']:.2f} | "
                f"mode={mode} v={linear_v:.2f} w={angular_v:.2f} obstacle={self.latest_penalty:.2f}"
            )

            self._draw_overlay(
                frame=frame,
                goal_text=current_goal,
                scores=scores,
                meta=meta,
                mode=mode,
                linear_v=linear_v,
                angular_v=angular_v,
            )

        except Exception as e:
            self.get_logger().error(f"inference/control 오류: {e}")

    def _draw_overlay(
        self,
        frame,
        goal_text,
        scores,
        meta,
        mode,
        linear_v,
        angular_v,
    ):
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.line(vis, (int(w * 0.3), 0), (int(w * 0.3), h), (100, 100, 255), 1)
        cv2.line(vis, (int(w * 0.4), 0), (int(w * 0.4), h), (255, 0, 0), 2)
        cv2.line(vis, (int(w * 0.6), 0), (int(w * 0.6), h), (255, 0, 0), 2)
        cv2.line(vis, (int(w * 0.7), 0), (int(w * 0.7), h), (100, 100, 255), 1)

        lines = [
            (f"Goal: {goal_text}", (0, 255, 0)),
            (f"State: {self.nav_state}   Mode: {mode}", (255, 255, 0)),
            (f"L:{scores['left']:.3f}  C:{scores['center']:.3f}  R:{scores['right']:.3f}", (0, 255, 255)),
            (f"Best raw: {meta['best_raw']}  smooth: {meta['best_smooth']}", (255, 255, 0)),
            (f"Score: {meta['best_score']:.3f}  Margin: {meta['margin']:.4f}", (255, 255, 0)),
            (f"Visible: {meta['target_visible']}  AreaRatio: {meta['approx_area_ratio']:.2f}", (0, 165, 255)),
            (f"Obstacle penalty: {self.latest_penalty:.2f}", (0, 165, 255)),
            (f"cmd_vel -> v:{linear_v:.2f}  w:{angular_v:.2f}", (255, 0, 255)),
        ]

        for i, (text, color) in enumerate(lines):
            cv2.putText(
                vis,
                text,
                (15, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("CLIP Navigator", vis)
        cv2.waitKey(1)

    def stop_robot(self):
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        time.sleep(0.3)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()