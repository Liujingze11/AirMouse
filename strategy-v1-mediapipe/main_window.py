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
