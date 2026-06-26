#!/usr/bin/env python3
"""AirMouse v1 — 空气鼠标：用摄像头识别手势控制光标"""

import sys
import signal
from PyQt6.QtWidgets import QApplication, QMessageBox

from airmouse.main_window import MainWindow


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
                "无法打开默认摄像头。\n请检查摄像头是否被占用或已连接。",
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
