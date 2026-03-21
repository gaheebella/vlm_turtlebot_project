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
from cv_bridge import CvBridge

import torch
import open_clip


class GazeboCLIPSubscriber(Node):
    def __init__(self):
        super().__init__('gazebo_clip_subscriber')

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(self.device)
        self.model.eval()

        # 프롬프트 파일 로드
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

        self.get_logger().info("Gazebo CLIP subscriber started.")
        self.get_logger().info("Subscribed topic: /rgbd_camera/image")

    def load_prompts(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            raise ValueError(f"Prompt file is empty: {file_path}")

        return prompts

    def image_callback(self, msg: Image):
        now = time.time()
        if now - self.last_debug_time >= self.debug_interval_sec:
            self.last_debug_time = now
            self.get_logger().info("Image received")

        # 1. ROS Image -> OpenCV(BGR)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 2. OpenCV(BGR) -> RGB -> PIL
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # 3. preprocess
        try:
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Preprocess failed: {e}")
            return

        # 4. CLIP inference
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()

            best_idx = int(np.argmax(probs))
            best_label = self.labels[best_idx]
            best_prob = float(probs[best_idx])

        except Exception as e:
            self.get_logger().error(f"CLIP inference failed: {e}")
            return

        # 5. 1초마다 결과 로그 출력
        if now - self.last_log_time >= self.log_interval_sec:
            self.last_log_time = now

            top_k = min(3, len(self.labels))
            top_indices = np.argsort(probs)[::-1][:top_k]

            result_text = " | ".join(
                [f"{self.labels[i]}: {probs[i]:.4f}" for i in top_indices]
            )

            self.get_logger().info(
                f"[Top-1] {best_label} ({best_prob:.4f}) | {result_text}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GazeboCLIPSubscriber()

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