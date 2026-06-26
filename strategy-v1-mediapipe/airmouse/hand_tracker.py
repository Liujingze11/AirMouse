"""HandTracker — MediaPipe Hands 封装，从摄像头读取帧并提取手部关键点"""

import cv2
import mediapipe as mp
import numpy as np
from .config import (
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
