import numpy as np
from sensor_msgs.msg import LaserScan


def get_obstacle_penalty(scan_msg: LaserScan, threshold_m: float = 0.6) -> float:
    ranges = np.array(scan_msg.ranges, dtype=np.float32)

    # NaN, inf 제거
    valid = np.isfinite(ranges)
    ranges = ranges[valid]

    if len(ranges) == 0:
        return 0.0

    # 매우 단순 버전: 전체 중 최소값 사용
    min_dist = np.min(ranges)

    if min_dist >= threshold_m:
        return 0.0

    penalty = (threshold_m - min_dist) / threshold_m
    return float(max(0.0, min(1.0, penalty)))