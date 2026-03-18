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

        # CLIP 모델 로드 (한 번만)
        loaded = load_clip_model()
        if len(loaded) == 4:
            self.model, self.preprocess, _, self.device = loaded
        elif len(loaded) == 3:
            self.model, self.preprocess, self.device = loaded
        else:
            raise ValueError("load_clip_model() 반환값 개수를 확인하세요.")

        # Navigator 생성 (한 번만)
        self.navigator = ClipDirectionNavigator(
            model=self.model,
            preprocess=self.preprocess,
            device=self.device,
            model_name="ViT-B-32",
            history_size=5,
            center_bias=0.015,
        )

        # 목표 텍스트
        self.goal_text = "a chair"
        self.goal_lock = threading.Lock()

        # 프레임 제어
        self.frame_count = 0
        self.infer_interval = 1  # 1이면 매 프레임 추론

        # 장애물 패널티
        self.latest_penalty = 0.0

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 카메라 구독
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos
        )

        # LiDAR 구독
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )

        # cmd_vel 퍼블리셔
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 목표 입력 스레드
        self.input_thread = threading.Thread(
            target=self._goal_input_loop,
            daemon=True
        )
        self.input_thread.start()

        self.get_logger().info(f"CLIP Navigator started | goal: '{self.goal_text}'")
        self.get_logger().info("터미널에서 새 목표 텍스트를 입력하면 즉시 변경됩니다.")

    # ─────────────────────────────────────────────
    # LiDAR 콜백
    # ─────────────────────────────────────────────
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        total = len(ranges)

        if total == 0:
            self.latest_penalty = 0.0
            return

        # 전방 ±20도
        front_count = max(1, int(total * 20 / 360))
        front = np.concatenate([ranges[-front_count:], ranges[:front_count]])
        front = front[np.isfinite(front)]

        if len(front) == 0:
            self.latest_penalty = 0.0
            return

        min_dist = float(np.min(front))
        threshold = 0.5  # 0.5m 이내면 장애물

        if min_dist < threshold:
            self.latest_penalty = (threshold - min_dist) / threshold

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.4
            self.cmd_pub.publish(twist)

            self.get_logger().warn(
                f"[AVOIDANCE] 장애물 dist={min_dist:.2f}m → 회피 중"
            )
        else:
            self.latest_penalty = 0.0

    # ─────────────────────────────────────────────
    # 목표 텍스트 런타임 변경
    # ─────────────────────────────────────────────
    def _goal_input_loop(self):
        while True:
            try:
                with self.goal_lock:
                    current_goal = self.goal_text

                new_goal = input(f"\n새 목표 입력 (현재: '{current_goal}'): ").strip()
                if new_goal:
                    with self.goal_lock:
                        self.goal_text = new_goal
                    self.get_logger().info(f"목표 변경 → '{new_goal}'")
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"goal input 오류: {e}")
                break

    # ─────────────────────────────────────────────
    # 카메라 콜백
    # ─────────────────────────────────────────────
    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.infer_interval != 0:
            return

        # 장애물 회피 중이면 CLIP 추론 스킵
        if self.latest_penalty > 0.0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return

        with self.goal_lock:
            current_goal = self.goal_text

        try:
            scores = self.navigator.compute_direction_scores(
                frame_bgr=frame,
                text_goal=current_goal
            )

            linear_v, angular_v, best_raw, best_smooth, margin, mode = self.navigator.decide_velocity(
                scores=scores,
                obstacle_penalty=self.latest_penalty
            )

            twist = Twist()
            twist.linear.x = float(linear_v)
            twist.angular.z = float(angular_v)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"[CLIP] goal='{current_goal}' | "
                f"L:{scores['left']:.3f} C:{scores['center']:.3f} R:{scores['right']:.3f} | "
                f"raw={best_raw} smooth={best_smooth} margin={margin:.4f} | "
                f"mode={mode} v={linear_v:.2f} w={angular_v:.2f}"
            )

            self._draw_overlay(
                frame=frame,
                scores=scores,
                best_raw=best_raw,
                best_smooth=best_smooth,
                margin=margin,
                mode=mode,
                linear_v=linear_v,
                angular_v=angular_v,
                goal_text=current_goal
            )

        except Exception as e:
            self.get_logger().error(f"inference/control 오류: {e}")

    # ─────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────
    def _draw_overlay(
        self,
        frame,
        scores,
        best_raw,
        best_smooth,
        margin,
        mode,
        linear_v,
        angular_v,
        goal_text
    ):
        vis = frame.copy()
        h, w = vis.shape[:2]

        # ROI 경계선
        cv2.line(vis, (int(w * 0.3), 0), (int(w * 0.3), h), (100, 100, 255), 1)
        cv2.line(vis, (int(w * 0.4), 0), (int(w * 0.4), h), (255, 0, 0), 2)
        cv2.line(vis, (int(w * 0.6), 0), (int(w * 0.6), h), (255, 0, 0), 2)
        cv2.line(vis, (int(w * 0.7), 0), (int(w * 0.7), h), (100, 100, 255), 1)

        lines = [
            (f"Goal: {goal_text}", (0, 255, 0)),
            (f"L:{scores['left']:.3f}  C:{scores['center']:.3f}  R:{scores['right']:.3f}", (0, 255, 255)),
            (f"Best raw: {best_raw}  smooth: {best_smooth}", (255, 255, 0)),
            (f"Margin: {margin:.4f}  Mode: {mode}", (255, 255, 0)),
            (f"Obstacle penalty: {self.latest_penalty:.2f}", (0, 165, 255)),
            (f"cmd_vel -> v:{linear_v:.2f}  w:{angular_v:.2f}", (255, 0, 255)),
        ]

        for i, (text, color) in enumerate(lines):
            cv2.putText(
                vis,
                text,
                (15, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("CLIP Navigator", vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 전 정지 명령
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        time.sleep(0.3)

        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()