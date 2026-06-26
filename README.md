# AirMouse 🖱️✋

**用摄像头识别手势来控制鼠标光标** —— 像空气一样无形、自然。

> 适用于 Linux 桌面环境。Python + MediaPipe + PyQt6。

## 功能

| 手势 | 动作 |
|------|------|
| ✊ 握拳 0.8 秒 | 启动 / 暂停 |
| 👆 食指指向 | 移动光标 |
| 🤏 拇指 + 食指捏合 | 左键点击 / 拖拽 |
| 🖕 拇指 + 中指捏合 | 右键点击 |

- **平滑稳定**：EMA 指数移动平均 + 死区过滤，消除抖动
- **混合映射**：微动精确、快动加速，像真实鼠标
- **防误触**：捏合需保持 3 帧、握拳需 0.8 秒，避免意外触发
- **实时反馈**：摄像头预览 + 手部骨骼 + 状态指示灯
- **系统托盘**：可最小化到后台，不占屏幕
- **灵敏度实时可调**：底部滑块 0.5x ~ 5.0x

## 环境要求

- Linux (X11)
- Python >= 3.10
- 摄像头 (笔记本内置即可)

## 安装

```bash
git clone https://github.com/Liujingze11/AirMouse.git
cd AirMouse/strategy-v1-mediapipe

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使用

```bash
python3 main.py
```

1. 启动后默认 **暂停**（🔴 已暂停）
2. 对着摄像头 ✊ **握拳 0.8 秒** → 激活（🟢 运行中）
3. 👆 **伸出食指** → 光标跟随指尖移动
4. 🤏 **拇指食指捏合** → 左键点击；捏住不放并移动 → 拖拽
5. 🖕 **拇指中指捏合** → 右键点击
6. 再次 ✊ 握拳 → 暂停

## 项目结构

```
strategy-v1-mediapipe/
├── main.py                  # 入口
├── requirements.txt         # 依赖清单
├── airmouse/                # 核心包
│   ├── config.py            # 配置常量（阈值、灵敏度等）
│   ├── hand_tracker.py      # MediaPipe 手部关键点识别
│   ├── gesture_recognizer.py # 手势分类 + 状态机
│   ├── cursor_controller.py  # 光标平滑控制 (EMA + 死区)
│   └── main_window.py       # PyQt6 GUI + 系统托盘
└── tests/                   # 单元测试
    ├── test_gesture_recognizer.py
    └── test_state_machine.py
```

```
Camera → HandTracker → GestureRecognizer → CursorController → pyautogui
                            ↕
                       MainWindow (GUI)
```

## 运行测试

```bash
python3 -m pytest tests/ -v
```

## 策略目录

`strategy-v1-mediapipe/` 是第一种实现方案。后续不同方案（不同追踪算法、不同映射策略等）各放独立目录，例如 `strategy-v2-dnn/`。

## 许可证

MIT
