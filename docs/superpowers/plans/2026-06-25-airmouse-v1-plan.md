# AirMouse v1 实现计划

> **对于 agentic 工作者:** 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务实现此计划。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标:** 在 Linux 上构建通过摄像头识别手势来控制鼠标光标、支持点击/拖拽/右键的空气鼠标应用。

**架构:** 4 个独立模块通过明确接口协作——HandTracker（MediaPipe 封装）、GestureRecognizer（纯手势计算）、CursorController（EMA+死区平滑+pyautogui）、MainWindow（PyQt6 GUI+托盘）。数据流：Camera → HandTracker → GestureRecognizer(+状态机) → CursorController → pyautogui。

**技术栈:** Python 3.10+, MediaPipe 0.10+, OpenCV 4.x, PyQt6, pyautogui

## 全局约束

- Python >= 3.10（使用当前环境 3.10.12）
- 所有阈值参数集中在 config.py，不硬编码
- 文件放置于 `strategy-v1-mediapipe/` 目录
- 遵循 TDD：先写测试、验证失败、再实现
- 每个任务以独立 commit 结束

---

### Task 1: 配置文件 + 依赖清单

**文件:**
- Create: `strategy-v1-mediapipe/config.py`
- Create: `strategy-v1-mediapipe/requirements.txt`

**接口:**
- 产出: `config.py` 中的所有常量，供后续所有模块引用

- [ ] **Step 1: 创建 requirements.txt**

```
opencv-python>=4.8
mediapipe>=0.10
pyautogui>=0.9
PyQt6>=6.5
```

- [ ] **Step 2: 创建 config.py**

```python
"""AirMouse v1 — 全局配置常量"""

# --- 摄像头 ---
CAMERA_ID = 0           # /dev/video0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# --- 手势识别阈值（归一化坐标，0-1）---
POINTING_FINGERTIP_ABOVE_KNUCKLE = 0.04   # 食指尖 y 必须小于指根 y 这么多
FINGER_CURLED_MAX_DIST = 0.15             # 弯曲手指尖端到指根的最大距离
PINCH_THUMB_INDEX_MAX_DIST = 0.05         # 拇指-食指捏合最大距离
PINCH_THUMB_MIDDLE_MAX_DIST = 0.06        # 拇指-中指捏合最大距离
FIST_MAX_DIST_TO_WRIST = 0.25             # 握拳时指尖到手腕的最大距离

# --- 握拳防抖 ---
FIST_HOLD_FRAMES = 24     # 持续多少帧才触发（@30fps = 0.8s）

# --- 点击防抖 ---
PINCH_HOLD_FRAMES = 3     # 捏合持续多少帧才触发 mousedown

# --- 光标控制 ---
DEFAULT_SENSITIVITY = 2.0
SENSITIVITY_MIN = 0.5
SENSITIVITY_MAX = 5.0
EMA_ALPHA = 0.4           # 指数移动平均平滑系数
DEADZONE_PX = 2           # 死区（像素）
SPEED_THRESHOLD = 150     # 加速阈值（px/s）
SPEED_BOOST = 1.5         # 加速倍率

# --- MediaPipe ---
MEDIAPIPE_MAX_HANDS = 1
MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# --- UI ---
WINDOW_DEFAULT_WIDTH = 960
WINDOW_DEFAULT_HEIGHT = 720
STATUS_BAR_HEIGHT = 40
```

- [ ] **Step 3: 验证 config.py 可以正常导入**

```bash
cd strategy-v1-mediapipe && python3 -c "import config; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add strategy-v1-mediapipe/config.py strategy-v1-mediapipe/requirements.txt
git commit -m "feat: add config and requirements for AirMouse v1"
```

---

### Task 2: HandTracker 模块

**文件:**
- Create: `strategy-v1-mediapipe/hand_tracker.py`

**接口:**
- 消费: `config.py` 中的摄像头和 MediaPipe 常量
- 产出:
  - `class HandTracker`: `__init__(camera_id=CAMERA_ID, max_hands=MEDIAPIPE_MAX_HANDS, detection_conf=MEDIAPIPE_DETECTION_CONFIDENCE, tracking_conf=MEDIAPIPE_TRACKING_CONFIDENCE)`
  - `get_frame() -> tuple[np.ndarray | None, list[Any] | None, str | None]`: 返回 `(frame, landmarks_list, handedness_label)`，无手时 landmarks_list 为 None
  - `release() -> None`: 释放摄像头
  - 属性 `width`, `height`: 帧尺寸

- [ ] **Step 1: 编写 hand_tracker.py**

```python
"""HandTracker — MediaPipe Hands 封装，从摄像头读取帧并提取手部关键点"""

import cv2
import mediapipe as mp
import numpy as np
from config import (
    CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT,
    MEDIAPIPE_MAX_HANDS, MEDIAPIPE_DETECTION_CONFIDENCE, MEDIAPIPE_TRACKING_CONFIDENCE,
)


class HandTracker:
    def __init__(
        self,
        camera_id: int = CAMERA_ID,
        max_hands: int = MEDIAPIPE_MAX_HANDS,
        detection_conf: float = MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_conf: float = MEDIAPIPE_TRACKING_CONFIDENCE,
    ):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_frame(self) -> tuple[np.ndarray | None, list | None, str | None]:
        """读取一帧并返回 (frame, landmarks, handedness_label)
        
        Returns:
            frame: RGB 格式的帧 (H, W, 3)，或 None（摄像头读取失败）
            landmarks: 归一化的关键点列表，每个元素有 .x .y .z 属性；无手时为 None
            handedness_label: 'Left' 或 'Right'，无手时为 None
        """
        ret, frame_bgr = self.cap.read()
        if not ret:
            return None, None, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            # 只取 confidence 最高的那只手
            best_idx = 0
            if len(results.multi_hand_landmarks) > 1:
                best_idx = max(
                    range(len(results.multi_handedness)),
                    key=lambda i: results.multi_handedness[i].classification[0].score,
                )

            landmarks = results.multi_hand_landmarks[best_idx]
            handedness = results.multi_handedness[best_idx].classification[0].label
            return frame_rgb, list(landmarks.landmark), handedness

        return frame_rgb, None, None

    def release(self):
        self.cap.release()
        self.hands.close()
```

- [ ] **Step 2: 手动验证 HandTracker 能读取摄像头**

```bash
cd strategy-v1-mediapipe && python3 -c "
from hand_tracker import HandTracker
import time
tracker = HandTracker()
for _ in range(10):
    frame, landmarks, hand = tracker.get_frame()
    if landmarks:
        print(f'检测到手: {hand}, 关键点数: {len(landmarks)}')
    else:
        print('无手')
    time.sleep(0.1)
tracker.release()
print('测试通过')
"
```

- [ ] **Step 3: Commit**

```bash
git add strategy-v1-mediapipe/hand_tracker.py
git commit -m "feat: add HandTracker module (MediaPipe + cv2 camera)"
```

---

### Task 3: GestureRecognizer 模块（手势枚举 + 静态识别函数）

**文件:**
- Create: `strategy-v1-mediapipe/gesture_recognizer.py`

**接口:**
- 消费: `config.py` 中的手势阈值常量
- 产出:
  - `class Gesture(enum.Enum)`: `NONE, POINTING, PINCH_THUMB_INDEX, PINCH_THUMB_MIDDLE, FIST`
  - `def classify_gesture(landmarks: list) -> Gesture`: 纯函数，从关键点列表返回手势类型

- [ ] **Step 1: 创建 gesture_recognizer.py**

```python
"""GestureRecognizer — 从 MediaPipe 手部关键点识别手势"""

from enum import Enum, auto
import math
from config import (
    POINTING_FINGERTIP_ABOVE_KNUCKLE,
    FINGER_CURLED_MAX_DIST,
    PINCH_THUMB_INDEX_MAX_DIST,
    PINCH_THUMB_MIDDLE_MAX_DIST,
    FIST_MAX_DIST_TO_WRIST,
)


class Gesture(Enum):
    NONE = auto()
    POINTING = auto()           # 食指指向（控制光标移动）
    PINCH_THUMB_INDEX = auto()  # 拇指+食指捏合（左键）
    PINCH_THUMB_MIDDLE = auto() # 拇指+中指捏合（右键）
    FIST = auto()               # 握拳（暂停/启动）


def _distance(lm_a, lm_b) -> float:
    """两个归一化关键点之间的欧几里得距离"""
    return math.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2 + (lm_a.z - lm_b.z)**2)


def classify_gesture(landmarks: list) -> Gesture:
    """输入 MediaPipe 归一化关键点列表 (21个)，返回手势类型
    
    landmarks[i] 有 .x .y .z 属性，值域均为 0-1
    """
    if landmarks is None or len(landmarks) < 21:
        return Gesture.NONE

    # 关键点索引
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_MCP = 5       # 食指根部
    MIDDLE_TIP = 12
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17

    lm = landmarks
    wrist = lm[WRIST]

    # --- 握拳: 所有指尖到手腕距离都很近 ---
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    all_fingers_close = all(
        _distance(lm[tip], wrist) < FIST_MAX_DIST_TO_WRIST
        for tip in tips
    )
    if all_fingers_close:
        return Gesture.FIST

    # --- 食指指向: 食指伸直（指尖高于指根），其他手指弯曲 ---
    index_pointing = (
        lm[INDEX_TIP].y < lm[INDEX_MCP].y - POINTING_FINGERTIP_ABOVE_KNUCKLE
    )
    middle_curled = (
        _distance(lm[MIDDLE_TIP], lm[MIDDLE_MCP]) < FINGER_CURLED_MAX_DIST
    )
    ring_curled = (
        _distance(lm[RING_TIP], lm[RING_MCP]) < FINGER_CURLED_MAX_DIST
    )
    pinky_curled = (
        _distance(lm[PINKY_TIP], lm[PINKY_MCP]) < FINGER_CURLED_MAX_DIST
    )
    if index_pointing and middle_curled and ring_curled and pinky_curled:
        # 检查拇指-食指是否捏合
        if _distance(lm[THUMB_TIP], lm[INDEX_TIP]) < PINCH_THUMB_INDEX_MAX_DIST:
            return Gesture.PINCH_THUMB_INDEX
        # 检查拇指-中指是否捏合（仅当中指也接近拇指时）
        if _distance(lm[THUMB_TIP], lm[MIDDLE_TIP]) < PINCH_THUMB_MIDDLE_MAX_DIST:
            return Gesture.PINCH_THUMB_MIDDLE
        return Gesture.POINTING

    # --- 非指向状态下也检查捏合 ---
    if _distance(lm[THUMB_TIP], lm[INDEX_TIP]) < PINCH_THUMB_INDEX_MAX_DIST:
        return Gesture.PINCH_THUMB_INDEX
    if _distance(lm[THUMB_TIP], lm[MIDDLE_TIP]) < PINCH_THUMB_MIDDLE_MAX_DIST:
        return Gesture.PINCH_THUMB_MIDDLE

    return Gesture.NONE


def get_fingertip_position(landmarks: list) -> tuple[float, float] | None:
    """返回食指尖的像素坐标 (x, y)，用于光标控制"""
    if landmarks is None or len(landmarks) < 21:
        return None
    INDEX_TIP = 8
    return (landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)
```

- [ ] **Step 2: 写测试验证手势逻辑**

```python
# strategy-v1-mediapipe/test_gesture_recognizer.py
"""GestureRecognizer 单元测试"""

import sys
sys.path.insert(0, '.')

from gesture_recognizer import Gesture, classify_gesture

# 模拟 MediaPipe landmark 对象
class FakeLM:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

def make_flat_hand():
    """模拟一张张开的手：指尖 y 小于指根 y（在屏幕上更高）"""
    return [
        FakeLM(0.5, 0.8, 0),   # 0: wrist
        FakeLM(0.45, 0.7, 0), FakeLM(0.43, 0.6, 0), FakeLM(0.41, 0.5, 0), FakeLM(0.39, 0.4, 0),  # thumb
        FakeLM(0.5, 0.7, 0),   # 5: index MCP
        FakeLM(0.51, 0.55, 0), FakeLM(0.52, 0.45, 0), FakeLM(0.53, 0.35, 0),  # index
        FakeLM(0.55, 0.7, 0),  # 9: middle MCP
        FakeLM(0.56, 0.55, 0), FakeLM(0.57, 0.45, 0), FakeLM(0.58, 0.35, 0),  # middle
        FakeLM(0.6, 0.7, 0),   # 13: ring MCP
        FakeLM(0.61, 0.55, 0), FakeLM(0.62, 0.45, 0), FakeLM(0.63, 0.35, 0),  # ring
        FakeLM(0.65, 0.7, 0),  # 17: pinky MCP
        FakeLM(0.66, 0.6, 0), FakeLM(0.67, 0.5, 0), FakeLM(0.68, 0.4, 0),   # pinky
    ]

def make_pointing_hand():
    """模拟食指指向：食指伸直，其他手指弯曲"""
    lms = []
    lms.append(FakeLM(0.5, 0.8, 0))    # 0: wrist
    # 拇指（弯曲靠近手掌）
    lms.append(FakeLM(0.48, 0.72, 0))
    lms.append(FakeLM(0.47, 0.74, 0))
    lms.append(FakeLM(0.46, 0.76, 0))
    lms.append(FakeLM(0.45, 0.75, 0))  # 4: thumb tip (靠近手掌)
    # 食指（伸直向上）
    lms.append(FakeLM(0.5, 0.65, 0))   # 5: index MCP
    lms.append(FakeLM(0.5, 0.5, 0))
    lms.append(FakeLM(0.5, 0.3, 0))
    lms.append(FakeLM(0.5, 0.1, 0))    # 8: index tip (高高在上)
    # 中指（弯曲，尖端靠近 MCP）
    lms.append(FakeLM(0.55, 0.65, 0))  # 9: middle MCP
    lms.append(FakeLM(0.58, 0.63, 0))
    lms.append(FakeLM(0.6, 0.62, 0))
    lms.append(FakeLM(0.62, 0.63, 0))  # 12: middle tip (靠近 MCP，弯曲)
    # 无名指（弯曲）
    lms.append(FakeLM(0.6, 0.65, 0))   # 13: ring MCP
    lms.append(FakeLM(0.63, 0.63, 0))
    lms.append(FakeLM(0.66, 0.62, 0))
    lms.append(FakeLM(0.69, 0.63, 0))  # 16: ring tip (靠近 MCP)
    # 小指（弯曲）
    lms.append(FakeLM(0.65, 0.65, 0))  # 17: pinky MCP
    lms.append(FakeLM(0.68, 0.63, 0))
    lms.append(FakeLM(0.71, 0.62, 0))
    lms.append(FakeLM(0.74, 0.63, 0))  # 20: pinky tip (靠近 MCP)
    return lms

def make_pinch_thumb_index():
    """模拟拇指+食指捏合"""
    lms = make_pointing_hand()
    # 把拇指尖移到食指尖旁边
    lms[4] = FakeLM(0.49, 0.11, 0)  # thumb tip 紧挨 index tip (0.5, 0.1)
    return lms

def make_pinch_thumb_middle():
    """模拟拇指+中指捏合"""
    lms = make_pointing_hand()
    # 拇指尖靠近中指尖(0.62, 0.63)
    lms[4] = FakeLM(0.61, 0.62, 0)
    return lms

def make_fist():
    """模拟握拳：所有指尖靠近手腕"""
    WRIST = FakeLM(0.5, 0.8, 0)
    return [
        WRIST,                                                        # 0
        FakeLM(0.49, 0.78, 0), FakeLM(0.48, 0.79, 0), FakeLM(0.47, 0.8, 0), FakeLM(0.46, 0.79, 0),  # thumb
        FakeLM(0.5, 0.78, 0),                                         # 5: index MCP
        FakeLM(0.48, 0.79, 0), FakeLM(0.47, 0.8, 0), FakeLM(0.46, 0.79, 0),  # index
        FakeLM(0.52, 0.78, 0),                                        # 9: middle MCP
        FakeLM(0.5, 0.79, 0), FakeLM(0.49, 0.8, 0), FakeLM(0.48, 0.79, 0),  # middle
        FakeLM(0.54, 0.78, 0),                                        # 13: ring MCP
        FakeLM(0.52, 0.79, 0), FakeLM(0.51, 0.8, 0), FakeLM(0.5, 0.79, 0),   # ring
        FakeLM(0.56, 0.78, 0),                                        # 17: pinky MCP
        FakeLM(0.54, 0.79, 0), FakeLM(0.53, 0.8, 0), FakeLM(0.52, 0.79, 0),  # pinky
    ]


def test_pointing():
    result = classify_gesture(make_pointing_hand())
    assert result == Gesture.POINTING, f"Expected POINTING, got {result}"

def test_pinch_index():
    result = classify_gesture(make_pinch_thumb_index())
    assert result == Gesture.PINCH_THUMB_INDEX, f"Expected PINCH_THUMB_INDEX, got {result}"

def test_pinch_middle():
    result = classify_gesture(make_pinch_thumb_middle())
    assert result == Gesture.PINCH_THUMB_MIDDLE, f"Expected PINCH_THUMB_MIDDLE, got {result}"

def test_fist():
    result = classify_gesture(make_fist())
    assert result == Gesture.FIST, f"Expected FIST, got {result}"

def test_flat_hand():
    """全张开的手不应被识别为特定手势（既非捏合也非握拳也非指向）"""
    result = classify_gesture(make_flat_hand())
    # 全张开可能是 NONE 或 POINTING（取决于食指是否明显高于指根）
    assert result not in [Gesture.PINCH_THUMB_INDEX, Gesture.PINCH_THUMB_MIDDLE, Gesture.FIST], \
        f"Flat hand should not be pinch or fist, got {result}"


if __name__ == '__main__':
    test_pointing()
    print("✓ test_pointing passed")
    test_pinch_index()
    print("✓ test_pinch_index passed")
    test_pinch_middle()
    print("✓ test_pinch_middle passed")
    test_fist()
    print("✓ test_fist passed")
    test_flat_hand()
    print("✓ test_flat_hand passed")
    print("\n所有测试通过!")
```

- [ ] **Step 3: 运行测试**

```bash
cd strategy-v1-mediapipe && python3 test_gesture_recognizer.py
```

- [ ] **Step 4: Commit**

```bash
git add strategy-v1-mediapipe/gesture_recognizer.py strategy-v1-mediapipe/test_gesture_recognizer.py
git commit -m "feat: add GestureRecognizer with gesture classification"
```

---

### Task 4: 手势状态机

**文件:**
- Modify: `strategy-v1-mediapipe/gesture_recognizer.py`（追加 GestureStateMachine 类）

**接口:**
- 消费: `Gesture` 枚举, `FIST_HOLD_FRAMES`, `PINCH_HOLD_FRAMES` 来自 config
- 产出:
  - `class AppState(enum.Enum)`: `PAUSED, ACTIVE, LEFT_DOWN, DRAGGING`
  - `class GestureStateMachine`: `__init__()`, `update(gesture: Gesture) -> AppState`

- [ ] **Step 1: 追加 AppState 和 GestureStateMachine 到 gesture_recognizer.py**

在 gesture_recognizer.py 末尾追加：

```python
from enum import Enum as _Enum, auto as _auto


class AppState(_Enum):
    PAUSED = _auto()     # 暂停，不控制光标
    ACTIVE = _auto()     # 运行中，食指指向=移动光标
    LEFT_DOWN = _auto()  # 左键按下
    DRAGGING = _auto()   # 拖拽中


class GestureStateMachine:
    """手势状态机：管理 App 状态转换，处理防抖"""

    def __init__(self):
        self.state = AppState.PAUSED          # 启动时默认暂停，握拳激活
        self._fist_counter = 0
        self._pinch_counter = 0
        self._right_click_fired = False       # 右键只触发一次，需松开后重置

    def update(self, gesture: Gesture) -> AppState:
        """输入当前帧手势，返回新的 AppState"""

        # --- 握拳切换 暂停/运行 ---
        if gesture == Gesture.FIST:
            self._fist_counter += 1
            if self._fist_counter >= FIST_HOLD_FRAMES:
                self._fist_counter = 0
                if self.state == AppState.PAUSED:
                    self.state = AppState.ACTIVE
                elif self.state == AppState.ACTIVE:
                    self.state = AppState.PAUSED
                    # 暂停时释放所有按键
                    self._reset_pinch()
        else:
            self._fist_counter = 0

        # 暂停状态不处理其他手势
        if self.state == AppState.PAUSED:
            return self.state

        # --- 右键点击（瞬时触发）---
        if gesture == Gesture.PINCH_THUMB_MIDDLE:
            if not self._right_click_fired:
                self._right_click_fired = True
                # 右键触发后回到 ACTIVE（右键是瞬时操作）
                self._reset_pinch()
                # 注意：实际 mouse 调用在 CursorController 中
                # 这里通过状态转换来通知
        else:
            self._right_click_fired = False

        # --- 左键捏合 ---
        if gesture == Gesture.PINCH_THUMB_INDEX:
            self._pinch_counter += 1
            if self._pinch_counter >= PINCH_HOLD_FRAMES:
                if self.state == AppState.ACTIVE:
                    self.state = AppState.LEFT_DOWN
                elif self.state == AppState.LEFT_DOWN:
                    self.state = AppState.DRAGGING
        else:
            # 捏合松开了
            if self.state in (AppState.LEFT_DOWN, AppState.DRAGGING):
                self.state = AppState.ACTIVE
            self._pinch_counter = 0

        return self.state

    def _reset_pinch(self):
        self._pinch_counter = 0

    @property
    def is_active(self) -> bool:
        return self.state != AppState.PAUSED
```

- [ ] **Step 2: 写状态机测试**

```python
# strategy-v1-mediapipe/test_state_machine.py
"""GestureStateMachine 单元测试"""

import sys
sys.path.insert(0, '.')

from gesture_recognizer import (
    Gesture, AppState, GestureStateMachine,
    FIST_HOLD_FRAMES, PINCH_HOLD_FRAMES,
)


def test_initial_state():
    sm = GestureStateMachine()
    assert sm.state == AppState.PAUSED, "Initial state should be PAUSED"


def test_fist_activates():
    sm = GestureStateMachine()
    # 发送 FIST_HOLD_FRAMES 帧握拳
    for _ in range(FIST_HOLD_FRAMES):
        state = sm.update(Gesture.FIST)
    assert state == AppState.ACTIVE, f"Should be ACTIVE after holding fist, got {state}"


def test_fist_pauses():
    sm = GestureStateMachine()
    # 先激活
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    assert sm.state == AppState.ACTIVE
    # 再握拳暂停
    for _ in range(FIST_HOLD_FRAMES):
        state = sm.update(Gesture.FIST)
    assert state == AppState.PAUSED, f"Should be PAUSED, got {state}"


def test_fist_interrupted():
    """握拳不够久不会触发"""
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES - 1):
        state = sm.update(Gesture.FIST)
    assert state == AppState.PAUSED, "Should stay PAUSED if fist not held long enough"


def test_pinch_left_down():
    sm = GestureStateMachine()
    # 激活
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    # 捏合防抖
    for _ in range(PINCH_HOLD_FRAMES):
        state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.LEFT_DOWN, f"Should be LEFT_DOWN, got {state}"


def test_pinch_release():
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    for _ in range(PINCH_HOLD_FRAMES):
        sm.update(Gesture.PINCH_THUMB_INDEX)
    assert sm.state == AppState.LEFT_DOWN
    # 松开捏合
    state = sm.update(Gesture.POINTING)
    assert state == AppState.ACTIVE, f"Should return to ACTIVE on release, got {state}"


def test_dragging():
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    for _ in range(PINCH_HOLD_FRAMES):
        sm.update(Gesture.PINCH_THUMB_INDEX)
    assert sm.state == AppState.LEFT_DOWN
    # 继续保持捏合
    state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.DRAGGING, f"Should be DRAGGING, got {state}"


def test_ignores_gestures_when_paused():
    sm = GestureStateMachine()
    # 暂停状态下，捏合不应该有反应
    for _ in range(PINCH_HOLD_FRAMES):
        state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.PAUSED, f"Should stay PAUSED, got {state}"


if __name__ == '__main__':
    test_initial_state()
    print("✓ test_initial_state passed")
    test_fist_activates()
    print("✓ test_fist_activates passed")
    test_fist_pauses()
    print("✓ test_fist_pauses passed")
    test_fist_interrupted()
    print("✓ test_fist_interrupted passed")
    test_pinch_left_down()
    print("✓ test_pinch_left_down passed")
    test_pinch_release()
    print("✓ test_pinch_release passed")
    test_dragging()
    print("✓ test_dragging passed")
    test_ignores_gestures_when_paused()
    print("✓ test_ignores_gestures_when_paused passed")
    print("\n所有状态机测试通过!")
```

- [ ] **Step 3: 运行测试**

```bash
cd strategy-v1-mediapipe && python3 test_state_machine.py
```

- [ ] **Step 4: Commit**

```bash
git add strategy-v1-mediapipe/gesture_recognizer.py strategy-v1-mediapipe/test_state_machine.py
git commit -m "feat: add GestureStateMachine for app state management"
```

---

### Task 5: CursorController 模块

**文件:**
- Create: `strategy-v1-mediapipe/cursor_controller.py`

**接口:**
- 消费: `config.py` 的光标控制常量, `Gesture` 和 `AppState` from gesture_recognizer
- 产出:
  - `class CursorController`: 
    - `__init__(sensitivity=DEFAULT_SENSITIVITY)`
    - `update(gesture: Gesture, app_state: AppState, fingertip_pos, frame_w, frame_h, dt) -> str | None`: 更新光标位置，返回事件描述字符串
    - `set_sensitivity(val: float) -> None`
    - `reset() -> None`: 重置平滑状态

- [ ] **Step 1: 创建 cursor_controller.py**

```python
"""CursorController — EMA+死区平滑 + 混合映射 + pyautogui 鼠标控制"""

import math
import time
import pyautogui
from config import (
    DEFAULT_SENSITIVITY, SENSITIVITY_MIN, SENSITIVITY_MAX,
    EMA_ALPHA, DEADZONE_PX, SPEED_THRESHOLD, SPEED_BOOST,
)
from gesture_recognizer import Gesture, AppState

# pyautogui 安全设置：鼠标移到屏幕角落不会触发异常（因为我们可能快速移动）
pyautogui.FAILSAFE = False


class CursorController:
    def __init__(self, sensitivity: float = DEFAULT_SENSITIVITY):
        self.sensitivity = sensitivity
        # EMA 状态
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None
        # 上一帧指尖位置（归一化坐标）
        self._prev_norm_x: float | None = None
        self._prev_norm_y: float | None = None
        # 时间
        self._prev_time: float = time.time()
        # 右键触发标记
        self._right_click_pending = False
        # 左键是否已按下
        self._left_down = False

    def set_sensitivity(self, val: float):
        self.sensitivity = max(SENSITIVITY_MIN, min(SENSITIVITY_MAX, val))

    def reset(self):
        self._smoothed_x = None
        self._smoothed_y = None
        self._prev_norm_x = None
        self._prev_norm_y = None

    def update(
        self,
        gesture: Gesture,
        app_state: AppState,
        fingertip_pos: tuple[float, float] | None,
        frame_w: int,
        frame_h: int,
    ) -> str | None:
        """
        每帧调用，处理光标移动和点击。
        
        Args:
            gesture: 当前帧手势
            app_state: 当前 App 状态
            fingertip_pos: 食指尖归一化坐标 (nx, ny)，或 None
            frame_w, frame_h: 帧尺寸
        
        Returns:
            事件描述字符串（用于 UI 状态显示），或 None
        """
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt <= 0:
            dt = 1.0 / 30.0

        event = None

        # --- 暂停状态：释放所有按键，不做任何操作 ---
        if app_state == AppState.PAUSED:
            if self._left_down:
                pyautogui.mouseUp()
                self._left_down = False
            self.reset()
            return "已暂停"

        if fingertip_pos is None:
            self.reset()
            return "无手"

        nx, ny = fingertip_pos

        # --- 处理右键（瞬时触发）---
        if gesture == Gesture.PINCH_THUMB_MIDDLE and not self._right_click_pending:
            self._right_click_pending = True
            pyautogui.rightClick()
            event = "右键点击"
        elif gesture != Gesture.PINCH_THUMB_MIDDLE:
            self._right_click_pending = False

        # --- 处理左键按下/拖拽/释放 ---
        if app_state == AppState.LEFT_DOWN and not self._left_down:
            pyautogui.mouseDown()
            self._left_down = True
            event = "左键按下"
        elif app_state == AppState.DRAGGING:
            if not self._left_down:
                pyautogui.mouseDown()
                self._left_down = True
            event = "拖拽中"
        elif app_state == AppState.ACTIVE and self._left_down:
            pyautogui.mouseUp()
            self._left_down = False
            event = "左键释放"

        # --- 光标移动（仅在 ACTIVE 状态或拖拽时移动）---
        if app_state in (AppState.ACTIVE, AppState.DRAGGING):
            if self._prev_norm_x is not None and self._prev_norm_y is not None:
                # 计算归一化位移
                dx_norm = nx - self._prev_norm_x
                dy_norm = ny - self._prev_norm_y

                # 转换为像素位移
                dx_px = dx_norm * frame_w * self.sensitivity
                dy_px = dy_norm * frame_h * self.sensitivity

                # EMA 平滑
                if self._smoothed_x is None:
                    self._smoothed_x = dx_px
                    self._smoothed_y = dy_px
                else:
                    self._smoothed_x = EMA_ALPHA * dx_px + (1 - EMA_ALPHA) * self._smoothed_x
                    self._smoothed_y = EMA_ALPHA * dy_px + (1 - EMA_ALPHA) * self._smoothed_y

                # 死区检查
                move_x = self._smoothed_x if abs(self._smoothed_x) >= DEADZONE_PX else 0
                move_y = self._smoothed_y if abs(self._smoothed_y) >= DEADZONE_PX else 0

                # 速度检测 → 加速
                speed = math.sqrt(dx_px**2 + dy_px**2) / max(dt, 0.001)
                if speed > SPEED_THRESHOLD:
                    move_x *= SPEED_BOOST
                    move_y *= SPEED_BOOST

                # 移动光标
                if move_x != 0 or move_y != 0:
                    pyautogui.moveRel(int(move_x), int(move_y), _pause=False)

            self._prev_norm_x = nx
            self._prev_norm_y = ny

            if event is None:
                if gesture == Gesture.POINTING:
                    event = "食指指向"
                elif gesture == Gesture.PINCH_THUMB_INDEX:
                    event = "捏合中"

        return event
```

- [ ] **Step 2: 验证模块导入**

```bash
cd strategy-v1-mediapipe && python3 -c "from cursor_controller import CursorController; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add strategy-v1-mediapipe/cursor_controller.py
git commit -m "feat: add CursorController with EMA smoothing and mouse control"
```

---

### Task 6: MainWindow GUI + 系统托盘

**文件:**
- Create: `strategy-v1-mediapipe/main_window.py`

**接口:**
- 消费: 所有前置模块
- 产出:
  - `class MainWindow(QMainWindow)`: 完整 GUI

- [ ] **Step 1: 创建 main_window.py**

```python
"""MainWindow — PyQt6 GUI：摄像头预览 + 手部骨骼 + 状态栏 + 系统托盘"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSystemTrayIcon, QMenu, QApplication,
    QStatusBar, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction, QPainter, QPen, QColor, QFont

from hand_tracker import HandTracker
from gesture_recognizer import (
    Gesture, AppState, classify_gesture, get_fingertip_position,
    GestureStateMachine,
)
from cursor_controller import CursorController
from config import (
    WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT,
    DEFAULT_SENSITIVITY,
)


# MediaPipe 手部骨骼连接线定义
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),             # 手掌横向连接
]


class CameraWidget(QLabel):
    """显示摄像头画面 + 手部骨骼叠加的 Widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setText("正在启动摄像头...")
        self.setStyleSheet("color: #888; font-size: 18px; background-color: #1a1a1a;")
        self._landmarks = None
        self._frame_w = 640
        self._frame_h = 480
        self._gesture_text = ""

    def set_frame(self, rgb_frame: np.ndarray, landmarks: list | None, gesture_text: str):
        """设置当前帧和关键点数据"""
        self._frame_w = rgb_frame.shape[1]
        self._frame_h = rgb_frame.shape[0]
        self._landmarks = landmarks
        self._gesture_text = gesture_text

        # numpy → QImage → QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 缩放以适应 widget
        scaled = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # 绘制骨骼和标注
        if landmarks:
            painter = QPainter(scaled)
            self._draw_landmarks(painter, scaled.width(), scaled.height())
            painter.end()

        self.setPixmap(scaled)

    def _draw_landmarks(self, painter: QPainter, w: int, h: int):
        """在 pixmap 上绘制手部骨骼和关键点"""
        # 骨骼连线
        pen = QPen(QColor(0, 255, 100, 180))
        pen.setWidth(2)
        painter.setPen(pen)
        for a, b in HAND_CONNECTIONS:
            if a < len(self._landmarks) and b < len(self._landmarks):
                x1 = int(self._landmarks[a].x * w)
                y1 = int(self._landmarks[a].y * h)
                x2 = int(self._landmarks[b].x * w)
                y2 = int(self._landmarks[b].y * h)
                painter.drawLine(x1, y1, x2, y2)

        # 关键点圆点
        painter.setBrush(QColor(255, 50, 50, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        for lm in self._landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            painter.drawEllipse(x - 3, y - 3, 6, 6)

        # 食指尖高亮（大圆点）
        if len(self._landmarks) > 8:
            ix = int(self._landmarks[8].x * w)
            iy = int(self._landmarks[8].y * h)
            painter.setBrush(QColor(0, 200, 255, 220))
            painter.drawEllipse(ix - 6, iy - 6, 12, 12)

        # 手势文字标注（右下角）
        if self._gesture_text:
            font = QFont("Sans", 18, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor(0, 255, 0, 200))
            painter.drawText(w - 200, h - 20, self._gesture_text)


class MainWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AirMouse v1")
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        # 核心模块
        self.tracker = HandTracker()
        self.gesture_sm = GestureStateMachine()
        self.cursor_ctrl = CursorController()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(1000 // 30)  # 30 fps

        # 构建 UI
        self._build_ui()
        self._build_tray()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 摄像头预览
        self.camera_widget = CameraWidget()
        layout.addWidget(self.camera_widget, 1)

        # 底部状态栏
        self._build_status_bar(layout)

    def _build_status_bar(self, layout):
        bar = QFrame()
        bar.setFixedHeight(50)
        bar.setStyleSheet("background-color: #2d2d2d; border-top: 1px solid #444;")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(10, 5, 10, 5)

        # 运行状态指示灯
        self.lbl_run = QLabel("🔴 已暂停")
        self.lbl_run.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        hbox.addWidget(self.lbl_run)

        hbox.addSpacing(10)
        self.lbl_hand = QLabel("❌ 无手")
        self.lbl_hand.setStyleSheet("color: #aaa; font-size: 14px;")
        hbox.addWidget(self.lbl_hand)

        hbox.addSpacing(10)
        self.lbl_gesture = QLabel("—")
        self.lbl_gesture.setStyleSheet("color: #aaa; font-size: 14px;")
        hbox.addWidget(self.lbl_gesture)

        hbox.addStretch()

        # 灵敏度滑块
        lbl_sens = QLabel("灵敏度:")
        lbl_sens.setStyleSheet("color: #ccc; font-size: 13px;")
        hbox.addWidget(lbl_sens)

        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(5, 50)  # 0.5 ~ 5.0, 以 0.1 为单位
        self.slider_sens.setValue(int(DEFAULT_SENSITIVITY * 10))
        self.slider_sens.setFixedWidth(120)
        self.slider_sens.valueChanged.connect(self._on_sensitivity_changed)
        hbox.addWidget(self.slider_sens)

        self.lbl_sens_val = QLabel(f"{DEFAULT_SENSITIVITY:.1f}")
        self.lbl_sens_val.setStyleSheet("color: #fff; font-size: 13px; min-width: 30px;")
        hbox.addWidget(self.lbl_sens_val)

        hbox.addSpacing(10)

        # 暂停/启动按钮
        self.btn_toggle = QPushButton("启动")
        self.btn_toggle.setFixedWidth(70)
        self.btn_toggle.setStyleSheet(
            "QPushButton { background: #4a4; color: white; border: none; "
            "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #5b5; }"
        )
        self.btn_toggle.clicked.connect(self._toggle_pause)
        hbox.addWidget(self.btn_toggle)

        # 最小化按钮
        btn_min = QPushButton("最小化")
        btn_min.setFixedWidth(70)
        btn_min.setStyleSheet(
            "QPushButton { background: #555; color: white; border: none; "
            "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #666; }"
        )
        btn_min.clicked.connect(self.hide)
        hbox.addWidget(btn_min)

        layout = self.centralWidget().layout()
        # 在 QVBoxLayout 中找到 camera_widget 的位置，添加 bar
        if isinstance(layout, QVBoxLayout):
            layout.addWidget(bar)

    def _build_tray(self):
        self.tray = QSystemTrayIcon(self)
        # 使用简单的彩色方块作为图标（因为没有图标文件）
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 180, 100))
        self.tray.setIcon(QIcon(pixmap))
        self.tray.setToolTip("AirMouse v1")

        menu = QMenu()
        action_show = QAction("显示窗口", self)
        action_show.triggered.connect(self.show_and_raise)
        menu.addAction(action_show)

        self.action_toggle_tray = QAction("暂停", self)
        self.action_toggle_tray.triggered.connect(self._toggle_pause)
        menu.addAction(self.action_toggle_tray)

        menu.addSeparator()
        action_quit = QAction("退出", self)
        action_quit.triggered.connect(self.close)
        menu.addAction(action_quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_and_raise()

    def _toggle_pause(self):
        if self.gesture_sm.state == AppState.PAUSED:
            self.gesture_sm.state = AppState.ACTIVE
            self.btn_toggle.setText("暂停")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background: #c44; color: white; border: none; "
                "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
                "QPushButton:hover { background: #d55; }"
            )
            self.action_toggle_tray.setText("暂停")
        else:
            self.gesture_sm.state = AppState.PAUSED
            self.btn_toggle.setText("启动")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background: #4a4; color: white; border: none; "
                "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
                "QPushButton:hover { background: #5b5; }"
            )
            self.action_toggle_tray.setText("启动")

    def _on_sensitivity_changed(self, val):
        sens = val / 10.0
        self.cursor_ctrl.set_sensitivity(sens)
        self.lbl_sens_val.setText(f"{sens:.1f}")

    def _process_frame(self):
        frame, landmarks, handedness = self.tracker.get_frame()

        if frame is None:
            return

        # 手势识别
        gesture = classify_gesture(landmarks)
        app_state = self.gesture_sm.update(gesture)

        # 光标控制
        fingertip = get_fingertip_position(landmarks)
        event = self.cursor_ctrl.update(
            gesture, app_state, fingertip,
            self.tracker.width, self.tracker.height,
        )

        # 更新 UI 状态
        self._update_status(app_state, landmarks is not None, gesture, event)

        # 更新摄像头预览
        gesture_text_map = {
            Gesture.POINTING: "👆 指向",
            Gesture.PINCH_THUMB_INDEX: "🤏 捏合",
            Gesture.PINCH_THUMB_MIDDLE: "🖕 右键",
            Gesture.FIST: "✊ 握拳",
            Gesture.NONE: "",
        }
        gesture_text = gesture_text_map.get(gesture, "")
        self.camera_widget.set_frame(frame, landmarks, gesture_text)

        # 同步按钮状态（当手势握拳切换时）
        if app_state == AppState.PAUSED:
            self.btn_toggle.setText("启动")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background: #4a4; color: white; border: none; "
                "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
                "QPushButton:hover { background: #5b5; }"
            )
            self.action_toggle_tray.setText("启动")
        else:
            self.btn_toggle.setText("暂停")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background: #c44; color: white; border: none; "
                "padding: 5px 12px; border-radius: 4px; font-size: 13px; }"
                "QPushButton:hover { background: #d55; }"
            )
            self.action_toggle_tray.setText("暂停")

    def _update_status(self, app_state, has_hand, gesture, event):
        # 运行状态
        if app_state == AppState.PAUSED:
            self.lbl_run.setText("🔴 已暂停")
            self.lbl_run.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        else:
            self.lbl_run.setText("🟢 运行中")
            self.lbl_run.setStyleSheet("color: #66ff66; font-weight: bold; font-size: 14px;")

        # 手检测
        if has_hand:
            self.lbl_hand.setText("✋ 有手")
            self.lbl_hand.setStyleSheet("color: #88ccff; font-size: 14px;")
        else:
            self.lbl_hand.setText("❌ 无手")
            self.lbl_hand.setStyleSheet("color: #aaa; font-size: 14px;")

        # 手势/事件
        status = event or "—"
        self.lbl_gesture.setText(status)
        self.lbl_gesture.setStyleSheet("color: #ffcc66; font-size: 14px;")

    def closeEvent(self, event):
        self.timer.stop()
        self.tracker.release()
        # pyautogui: 确保鼠标按键释放
        import pyautogui
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
        self.closed.emit()
        super().closeEvent(event)
```

- [ ] **Step 2: Commit**

```bash
git add strategy-v1-mediapipe/main_window.py
git commit -m "feat: add MainWindow GUI with camera preview and system tray"
```

---

### Task 7: 入口文件 main.py

**文件:**
- Create: `strategy-v1-mediapipe/main.py`

- [ ] **Step 1: 创建 main.py**

```python
#!/usr/bin/env python3
"""AirMouse v1 — 空气鼠标：用摄像头识别手势控制光标"""

import sys
import signal
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from main_window import MainWindow


def main():
    # 允许 Ctrl+C 退出
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("AirMouse")
    app.setQuitOnLastWindowClosed(False)  # 关闭窗口时隐藏到托盘

    # 检查摄像头
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(
                None, "摄像头错误",
                "无法打开摄像头 (/dev/video0)。\n请检查摄像头是否被占用或已连接。",
            )
            sys.exit(1)
        cap.release()
    except Exception as e:
        QMessageBox.critical(None, "错误", f"初始化摄像头失败: {e}")
        sys.exit(1)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试启动**

```bash
cd strategy-v1-mediapipe && timeout 3 python3 main.py 2>&1 || true
```

- [ ] **Step 3: Commit**

```bash
git add strategy-v1-mediapipe/main.py
git commit -m "feat: add main entry point"
```

---

### Task 8: 集成测试与最终验证

- [ ] **Step 1: 检查所有文件存在**

```bash
ls -la strategy-v1-mediapipe/*.py
```

- [ ] **Step 2: 运行单元测试**

```bash
cd strategy-v1-mediapipe && python3 test_gesture_recognizer.py && python3 test_state_machine.py
```

- [ ] **Step 3: 验证 import 链**

```bash
cd strategy-v1-mediapipe && python3 -c "
from config import *
from hand_tracker import HandTracker
from gesture_recognizer import Gesture, AppState, classify_gesture, get_fingertip_position, GestureStateMachine
from cursor_controller import CursorController
from main_window import MainWindow
print('所有模块导入成功!')
"
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: final integration verification"
```
