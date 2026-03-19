import numpy as np
from sensor_msgs.msg import LaserScan


def get_obstacle_penalty(scan_msg: LaserScan, threshold_m: float = 0.6) -> float:
    ranges = np.array(scan_msg.ranges, dtype=np.float32)
    total = len(ranges)

    if total == 0:
        return 0.0

    # 전방 ±20도만 확인
    front_angle = 20
    front_count = int(total * front_angle / 360)

    front = np.concatenate([ranges[-front_count:], ranges[:front_count]])
    front = front[np.isfinite(front)]

    if len(front) == 0:
        return 0.0

    min_dist = np.min(front)

    if min_dist >= threshold_m:
        return 0.0

    penalty = (threshold_m - min_dist) / threshold_m
    return float(max(0.0, min(1.0, penalty)))