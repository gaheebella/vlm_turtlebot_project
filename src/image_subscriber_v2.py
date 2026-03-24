"""
image_subscriber_v2.py

업그레이드된 메인 ROS2 노드
- CLIP  → 방향 탐색 (기본)
- YOLO  → 목표물 정밀 탐지 및 추적
- Depth Anything → 목표까지 거리 추정
- LiDAR → 물리적 장애물 긴급 회피
- LLM   → 자연어 명령 파싱

실행 방법:
    python3 image_subscriber_v2.py

모드 설정 (상단 CONFIG에서 변경):
    USE_YOLO   = True/False
    USE_DEPTH  = True/False
    USE_LLM    = True/False
"""

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

from clip_navigator import compute_direction_scores, decide_velocity
from clip_core import load_clip_model

# ── 사용할 모듈 ON/OFF ────────────────────────────────────
USE_YOLO  = True   # YOLOv8 목표 탐지
USE_DEPTH = False  # Depth Anything (라즈베리파이에서 느릴 수 있음)
USE_LLM   = True   # LLM 자연어 파싱
# ─────────────────────────────────────────────────────────

if USE_YOLO:
    from yolo_detector import detect_goal, decide_velocity_yolo, draw_detection

if USE_DEPTH:
    from depth_estimator import estimate_depth, get_depth_velocity

if USE_LLM:
    from llm_parser import parse_goal, parse_goal_simple


class ImageSubscriberV2(Node):
    def __init__(self):
        super().__init__('image_subscriber_v2')

        self.bridge = CvBridge()

        # ── CLIP 모델 로드 ──────────────────────────────
        self.get_logger().info("CLIP 모델 로딩 중...")
        loaded = load_clip_model()
        if len(loaded) == 4:
            self.model, self.preprocess, _, self.device = loaded
        else:
            self.model, self.preprocess, self.device = loaded
        self.get_logger().info("CLIP 모델 로딩 완료")

        # ── 상태 변수 ───────────────────────────────────
        self.goal_text  = "a chair"   # 현재 목표
        self.yolo_label = "chair"     # YOLO용 레이블 (COCO 클래스명)
        self.frame_count = 0
        self.latest_penalty = 0.0
        self.safe_to_forward = False

        # ── QoS 설정 ────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── ROS2 구독/발행 ──────────────────────────────
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── 목표 입력 스레드 ────────────────────────────
        self.input_thread = threading.Thread(
            target=self._goal_input_loop, daemon=True
        )
        self.input_thread.start()

        self.get_logger().info(
            f"ImageSubscriberV2 시작 | goal='{self.goal_text}' | "
            f"YOLO={USE_YOLO} DEPTH={USE_DEPTH} LLM={USE_LLM}"
        )

    # ── LiDAR 콜백 ──────────────────────────────────────
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        total = len(ranges)
        if total == 0:
            return

        front_count = int(total * 20 / 360)
        front = np.concatenate([ranges[-front_count:], ranges[:front_count]])
        front = front[np.isfinite(front)]

        if len(front) == 0:
            self.latest_penalty = 0.0
            return

        min_dist = float(np.min(front))
        threshold = 0.5
        safe_dist = 0.8

        if min_dist < threshold:
            self.latest_penalty = (threshold - min_dist) / threshold
            self.safe_to_forward = False

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.4
            self.cmd_pub.publish(twist)

            self.get_logger().warn(
                f"[AVOIDANCE] 장애물 dist={min_dist:.2f}m → 회피 중"
            )
        elif min_dist >= safe_dist:
            self.latest_penalty = 0.0
            self.safe_to_forward = True
        else:
            self.latest_penalty = 0.0
            self.safe_to_forward = False

    # ── 목표 텍스트 입력 ────────────────────────────────
    def _goal_input_loop(self):
        while True:
            try:
                user_input = input(
                    f"\n명령 입력 (현재: '{self.goal_text}'): "
                ).strip()

                if not user_input:
                    continue

                if USE_LLM:
                    # LLM으로 자연어 파싱 시도
                    parsed = parse_goal(user_input)
                    if not parsed:
                        parsed = parse_goal_simple(user_input)
                else:
                    # LLM 없으면 직접 입력
                    parsed = parse_goal_simple(user_input)

                self.goal_text = parsed

                # YOLO용 레이블 추출 (관사 제거)
                self.yolo_label = parsed.replace("a ", "").replace("an ", "").replace("the ", "").split()[0]

                self.get_logger().info(
                    f"목표 변경 → '{self.goal_text}' (YOLO: '{self.yolo_label}')"
                )

            except EOFError:
                break

    # ── 카메라 콜백 ─────────────────────────────────────
    def image_callback(self, msg):
        self.frame_count += 1

        # 장애물 회피 중이면 스킵
        if self.latest_penalty > 0.0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return

        try:
            linear_v = 0.0
            angular_v = 0.0
            mode = "IDLE"
            vis = frame.copy()

            # ── YOLO 탐지 (매 3프레임) ──────────────────
            if USE_YOLO and self.frame_count % 3 == 0:
                detection = detect_goal(frame, self.yolo_label)
                linear_v, angular_v, mode = decide_velocity_yolo(detection)
                vis = draw_detection(vis, detection, self.yolo_label)

                # Depth로 속도 조정 (매 10프레임)
                if USE_DEPTH and detection["detected"] and self.frame_count % 10 == 0:
                    depth_val = estimate_depth(frame, detection["box"])
                    depth_v, depth_mode = get_depth_velocity(depth_val)
                    # YOLO 속도와 Depth 속도 중 작은 값 사용 (안전)
                    linear_v = min(linear_v, depth_v)
                    mode = f"{mode}+{depth_mode}"

                    cv2.putText(vis, f"Depth: {depth_val:.1f}", (15, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            # ── YOLO 미사용 또는 미탐지 시 CLIP 폴백 ───
            elif self.frame_count % 1 == 0:
                scores = compute_direction_scores(
                    frame_bgr=frame,
                    text_goal=self.goal_text,
                    model=self.model,
                    preprocess=self.preprocess,
                    device=self.device,
                )
                linear_v, angular_v, best_name, margin, mode = decide_velocity(
                    scores=scores,
                    obstacle_penalty=0.0,
                )

                # CLIP 오버레이
                h, w = vis.shape[:2]
                cv2.line(vis, (int(w*0.3), 0), (int(w*0.3), h), (100,100,255), 1)
                cv2.line(vis, (int(w*0.4), 0), (int(w*0.4), h), (255,0,0), 2)
                cv2.line(vis, (int(w*0.6), 0), (int(w*0.6), h), (255,0,0), 2)
                cv2.line(vis, (int(w*0.7), 0), (int(w*0.7), h), (100,100,255), 1)

                cv2.putText(vis,
                    f"CLIP L:{scores['left']:.3f} C:{scores['center']:.3f} R:{scores['right']:.3f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # ── 전진 성분 추가 ──────────────────────────
            if self.safe_to_forward and linear_v == 0.0 and mode not in ("GOAL_REACHED",):
                linear_v = 0.08
                mode = mode + "+FWD"

            # ── cmd_vel 발행 ────────────────────────────
            twist = Twist()
            twist.linear.x = float(linear_v)
            twist.angular.z = float(angular_v)
            self.cmd_pub.publish(twist)

            # ── 공통 오버레이 ───────────────────────────
            cv2.putText(vis, f"Goal: {self.goal_text}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(vis, f"Mode: {mode}  v:{linear_v:.2f} w:{angular_v:.2f}",
                        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(vis, f"Obstacle: {self.latest_penalty:.2f}  Safe: {self.safe_to_forward}",
                        (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            cv2.imshow("CLIP Navigator V2", vis)
            cv2.waitKey(1)

            self.get_logger().info(
                f"goal='{self.goal_text}' | mode={mode} | "
                f"v={linear_v:.2f} w={angular_v:.2f} | "
                f"obstacle={self.latest_penalty:.2f}"
            )

        except Exception as e:
            self.get_logger().error(f"오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriberV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        time.sleep(0.3)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()