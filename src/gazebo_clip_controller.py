#!/usr/bin/env python3
import time
from typing import List

import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import torch
import open_clip


class GazeboCLIPController(Node):
    def __init__(self):
        super().__init__('gazebo_clip_controller')

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            Image,
            '/rgbd_camera/image',
            self.image_callback,
            qos
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(self.device)
        self.model.eval()

        self.labels = self.load_prompts("prompts/gazebo_basic.txt")
        self.get_logger().info(f"Loaded prompts: {self.labels}")

        with torch.no_grad():
            text_tokens = self.tokenizer(self.labels).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.last_log_time = 0.0
        self.log_interval_sec = 1.0
        self.last_debug_time = 0.0
        self.debug_interval_sec = 1.0

        # 제어 안정화용 파라미터
        self.conf_threshold = 0.55
        self.margin_threshold = 0.08

        self.get_logger().info("Gazebo CLIP controller started.")
        self.get_logger().info("Subscribed topic: /rgbd_camera/image")
        self.get_logger().info("Publishing control to: /cmd_vel")

    def load_prompts(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            raise ValueError(f"Prompt file is empty: {file_path}")

        return prompts

    def publish_stop(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def publish_forward(self):
        msg = Twist()
        msg.linear.x = 0.25
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def publish_turn_left(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.6
        self.cmd_pub.publish(msg)

    def publish_turn_right(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = -0.6
        self.cmd_pub.publish(msg)

    def image_callback(self, msg: Image):
        now = time.time()
        if now - self.last_debug_time >= self.debug_interval_sec:
            self.last_debug_time = now
            self.get_logger().info("Image received")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        try:
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Preprocess failed: {e}")
            return

        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()

            sorted_indices = np.argsort(probs)[::-1]
            best_idx = int(sorted_indices[0])
            second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else best_idx

            best_label = self.labels[best_idx]
            best_prob = float(probs[best_idx])
            second_prob = float(probs[second_idx])
            margin = best_prob - second_prob

        except Exception as e:
            self.get_logger().error(f"CLIP inference failed: {e}")
            return

        # -----------------------------
        # CLIP -> 제어 정책
        # -----------------------------
        action = "stop"

        if best_prob < self.conf_threshold or margin < self.margin_threshold:
            self.publish_stop()
            action = "stop (low confidence)"
        else:
            if best_label == "a red box":
                self.publish_forward()
                action = "forward"
            elif best_label == "a traffic cone":
                self.publish_turn_left()
                action = "turn_left"
            else:
                self.publish_stop()
                action = "stop (unknown label)"

        if now - self.last_log_time >= self.log_interval_sec:
            self.last_log_time = now

            top_k = min(3, len(self.labels))
            top_indices = np.argsort(probs)[::-1][:top_k]

            result_text = " | ".join(
                [f"{self.labels[i]}: {probs[i]:.4f}" for i in top_indices]
            )

            self.get_logger().info(
                f"[Top-1] {best_label} ({best_prob:.4f}) | "
                f"margin={margin:.4f} | action={action} | {result_text}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GazeboCLIPController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()