# 현재: 전체 scan 중 최솟값 → 옆에 벽 있어도 멈춰버림
min_dist = np.min(ranges)

# 수정: 전방 ±20도만 봐야 함
def get_obstacle_penalty(scan_msg: LaserScan, threshold_m: float = 0.6) -> float:
    ranges = np.array(scan_msg.ranges, dtype=np.float32)
    total = len(ranges)
    
    # 전방 ±20도 인덱스 계산
    front_angle = 20
    front_count = int(total * front_angle / 360)
    
    front = np.concatenate([ranges[-front_count:], ranges[:front_count]])
    front = front[np.isfinite(front)]
    
    if len(front) == 0:
        return 0.0
    
    min_dist = np.min(front)
    if min_dist >= threshold_m:
        return 0.0
    
    return float(max(0.0, min(1.0, (threshold_m - min_dist) / threshold_m)))