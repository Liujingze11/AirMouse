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
