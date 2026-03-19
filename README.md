# 📌 CLIP-based Semantic Navigation System
> CLIP을 활용하여 자연어 목표를 인식하고, 상태 기반 제어를 통해 로봇이 목표 객체를 탐색하고 접근하는 semantic navigation 시스템

## 1. Overview

본 프로젝트는 **자연어 기반 목표 지정이 가능한 자율주행 시스템**을 구현하는 것을 목표로 한다.
기존의 좌표 기반 또는 rule-based navigation과 달리, 사용자가 입력한 **텍스트 목표 (e.g., “a chair”)를 기반으로 로봇이 목표 객체를 탐색하고 접근**한다.

이를 위해 **CLIP 기반 이미지-텍스트 유사도**를 활용하여 perception과 control을 직접 연결하는 **semantic navigation pipeline**을 설계하였다.

---

## 2. Objective

* 자연어 기반 목표 설정 (Language-conditioned navigation)
* Perception → Control 직접 연결 (End-to-End 구조)
* 불확실한 환경에서의 목표 탐색 및 접근
* ROS2 기반 실시간 로봇 제어 시스템 구현

---

## 3. Key Idea

> “이미지를 이해해서 움직인다”가 아니라
> **“텍스트 목표와 가장 유사한 방향으로 움직인다”**

* 카메라 이미지 → 좌/중/우 분할
* 각 영역과 텍스트 간 CLIP similarity 계산
* 가장 유사한 방향으로 로봇을 이동

---

## 4. System Architecture

```
Camera (/image_raw)
        ↓
OpenCV (frame processing)
        ↓
CLIP (image-text similarity)
        ↓
Direction Scoring (Left / Center / Right)
        ↓
State Machine (SEARCH → ALIGN → APPROACH)
        ↓
Velocity Command (/cmd_vel)
        ↓
Robot Movement
```

---

## 5. System Flow

### ① Perception

* 입력: RGB 이미지
* 처리:

  * 좌 / 중 / 우 영역으로 crop
  * 각 영역에 대해 CLIP embedding 생성
  * 텍스트 embedding과 cosine similarity 계산

출력:

```python
{
  "left": score,
  "center": score,
  "right": score
}
```

---

### ② Direction Decision

* 가장 높은 score → 이동 방향 결정
* margin (1등 - 2등) → 신뢰도 판단
* center bias 적용 → 직진 안정성 확보

---

### ③ State Machine

| State       | 설명         |
| ----------- | ---------- |
| SEARCHING   | 목표 탐색 (회전) |
| ALIGNING    | 목표 방향 정렬   |
| APPROACHING | 목표로 전진     |
| AVOIDING    | 장애물 회피     |
| STOPPED     | 목표 도달      |

---

### ④ Control Logic

* 목표 미탐지 → 회전 탐색
* 목표 좌/우 → 회전 정렬
* 목표 정면 → 전진
* 장애물 감지 → CLIP 무시하고 회피

---

## 6. Obstacle Avoidance

* LiDAR `/scan` 사용
* 전방 거리 < 0.5m → 즉시 회피
* CLIP 제어보다 **우선순위 높음**

---

## 7. Implementation Details

### ✔ Tech Stack

* ROS2 (Humble)
* Python
* OpenCV
* OpenCLIP (ViT-B-32)
* CvBridge

---

### ✔ 주요 모듈

| 파일                    | 역할              |
| --------------------- | --------------- |
| `image_subscriber.py` | ROS2 노드, 전체 제어  |
| `clip_navigator.py`   | 방향 판단 및 상태 머신   |
| `clip_core.py`        | CLIP 모델 로딩      |
| `obstacle_avoider.py` | LiDAR 기반 장애물 감지 |

---

## 8. Results

### ✔ 동작 방식

* 로봇은 주변을 회전하며 목표 탐색
* 특정 프레임에서 목표 인식 시 해당 방향으로 이동
* 목표를 잃으면 다시 탐색 반복

---

### ✔ 특징

* 자연어 기반 목표 설정 가능
* 지도 없이 navigation 수행 (Map-free)
* perception → control 직접 연결

---

### ⚠ Limitations

* CLIP은 frame-based 모델 → temporal consistency 부족
* 동일 객체를 지속적으로 추적하지 못함
* detection 결과가 프레임마다 변동 (stochastic behavior)

---

## 9. Key Insight

> 본 시스템은 “지속 추적 기반 navigation”이 아니라
> **“프레임 단위 semantic detection 기반 navigation”**

---

## 10. Future Work

* Temporal smoothing (EMA / LSTM)
* Object tracking (persistent memory)
* Multi-frame confidence aggregation
* Depth estimation 기반 거리 인식
* VLA (Vision-Language-Action) 확장

---

## 11. Demo Description (발표용 한 줄)

> CLIP 기반 semantic perception을 활용하여 자연어 목표를 탐색하고, 상태 기반 제어를 통해 목표 객체로 접근하는 navigation 시스템입니다.

---

## 12. Contribution

* CLIP 기반 방향 추론 구조 설계
* State machine 기반 navigation 로직 구현
* ROS2 실시간 제어 시스템 구축
* Perception–Control 통합 구조 설계

> CLIP을 활용하여 자연어 목표를 인식하고, 상태 기반 제어를 통해 로봇이 목표 객체를 탐색하고 접근하는 semantic navigation 시스템



2026.03.19 업데이트
