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
