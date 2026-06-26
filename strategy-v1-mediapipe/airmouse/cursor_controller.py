"""CursorController — EMA+死区平滑 + 混合映射 + pyautogui 鼠标控制"""

import math
import time
import pyautogui
from .config import (
    DEFAULT_SENSITIVITY, SENSITIVITY_MIN, SENSITIVITY_MAX,
    EMA_ALPHA, DEADZONE_PX, SPEED_THRESHOLD, SPEED_BOOST,
)
from .gesture_recognizer import Gesture, AppState

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
        self._right_click_pending = False
        self._left_down = False

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
