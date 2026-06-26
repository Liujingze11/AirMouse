"""GestureRecognizer — 从 MediaPipe 手部关键点识别手势"""

from enum import Enum, auto
import math
from .config import (
    POINTING_FINGERTIP_ABOVE_KNUCKLE,
    FINGER_CURLED_MAX_DIST,
    PINCH_THUMB_INDEX_MAX_DIST,
    PINCH_THUMB_MIDDLE_MAX_DIST,
    FIST_MAX_DIST_TO_WRIST,
    FIST_HOLD_FRAMES,
    PINCH_HOLD_FRAMES,
)


# 关键点索引
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17


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
    """返回食指尖的归一化坐标 (x, y)，值域 0-1，用于光标控制"""
    if landmarks is None or len(landmarks) < 21:
        return None
    return (landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)


# ─── App State & State Machine ────────────────────────────────────────────────


class AppState(Enum):
    PAUSED = auto()     # 暂停，不控制光标
    ACTIVE = auto()     # 运行中，食指指向=移动光标
    LEFT_DOWN = auto()  # 左键按下
    DRAGGING = auto()   # 拖拽中


class GestureStateMachine:
    """手势状态机：管理 App 状态转换，处理防抖"""

    def __init__(self):
        self.state = AppState.PAUSED          # 启动时默认暂停，握拳激活
        self._fist_counter = 0
        self._pinch_counter = 0
        # 右键只触发一次，需松开后重置（在 CursorController 中处理）

    def update(self, gesture: Gesture) -> AppState:
        """输入当前帧手势，返回新的 AppState"""

        # --- 握拳切换 暂停/运行 ---
        if gesture == Gesture.FIST:
            # 握拳时如果正在左键按下/拖拽，先释放
            if self.state in (AppState.LEFT_DOWN, AppState.DRAGGING):
                self._reset_pinch()
                self.state = AppState.ACTIVE
                self._fist_counter = 0
                return self.state
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

        # --- 右键点击（瞬时触发，实际鼠标调用在 CursorController 中）---
        if gesture == Gesture.PINCH_THUMB_MIDDLE:
            self._reset_pinch()
            # 右键通过 CursorController 直接处理（基于原始手势判断）

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

    def toggle_pause(self) -> AppState:
        """手动切换暂停/运行状态，正确重置内部计数器"""
        self._reset_pinch()
        self._fist_counter = 0
        if self.state == AppState.PAUSED:
            self.state = AppState.ACTIVE
        else:
            self.state = AppState.PAUSED
        return self.state

    def _reset_pinch(self):
        self._pinch_counter = 0

    @property
    def is_active(self) -> bool:
        return self.state != AppState.PAUSED
