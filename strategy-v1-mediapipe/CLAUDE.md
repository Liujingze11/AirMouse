# strategy-v1-mediapipe — 项目上下文

> MediaPipe 手部关键点方案。属于 [AirMouse](../README.md) 多方案实验项目。
> 最后更新：2026-06-26

---

## 一、方案一句话

**摄像头 + MediaPipe 识别单手 21 个关键点 → 手势分类 → 控制 Linux 桌面光标。** 食指移动、捏合点击/拖拽、握拳暂停。

---

## 二、当前状态

- **v1 完成**（2026-06-26），测试全部通过，代码已重构为 `airmouse/` 包。
- **无阻塞问题**。

### 已知坑

| 项 | 说明 |
|----|------|
| Wayland | pyautogui 在 Wayland 上功能受限，仅 X11 完整支持 |
| 单手 | 只跟踪 confidence 最高的手 |
| 多显示器 | 只映射主显示器 |

### 下一步（v2 可选）

双手支持、自定义手势配置化、多屏适配、开机自启。

---

## 三、项目结构

```
strategy-v1-mediapipe/
├── CLAUDE.md                    # 本文件
├── README.md                    # 用户文档
├── docs/
│   └── design.md                # 设计文档（架构、手势规则、UI 设计）
├── main.py                      # 入口
├── requirements.txt
├── airmouse/                    # 核心包
│   ├── config.py                # 所有阈值、灵敏度、EMA 参数
│   ├── hand_tracker.py          # cv2 摄像头 + MediaPipe Hands → 21 关键点
│   ├── gesture_recognizer.py    # Gesture 分类 + AppState 状态机
│   ├── cursor_controller.py     # EMA+死区平滑 + 混合映射 + pyautogui
│   └── main_window.py           # PyQt6 GUI：预览 + 骨骼 + 状态栏 + 托盘
└── tests/
    ├── test_gesture_recognizer.py  # 6 个手势分类测试
    └── test_state_machine.py       # 8 个状态机测试
```

---

## 四、架构速览

```
Camera → HandTracker → GestureRecognizer(+状态机) → CursorController → pyautogui
                              ↕
                        MainWindow (GUI)
```

- **HandTracker** — 摄像头取帧 + MediaPipe 推理，返回 21 个归一化关键点
- **GestureRecognizer** — 纯函数 `classify_gesture(landmarks) → Gesture` + `GestureStateMachine` 管理 AppState（PAUSED/ACTIVE/LEFT_DOWN/DRAGGING）
- **CursorController** — EMA 平滑 + 死区 + 混合映射 + pyautogui 调用
- **MainWindow** — QTimer 30fps 驱动全流程，摄像头预览 + 骨骼叠加 + 系统托盘

> 手势规则、状态机图、UI 布局详见 `docs/design.md`
> 所有配置常量详见 `airmouse/config.py`

---

## 五、常用命令

```bash
# 运行
python3 main.py

# 测试
python3 -m pytest tests/ -v

# 导入验证
python3 -c "
from airmouse.config import *
from airmouse.hand_tracker import HandTracker
from airmouse.gesture_recognizer import Gesture, AppState, classify_gesture, GestureStateMachine
from airmouse.cursor_controller import CursorController
from airmouse.main_window import MainWindow
print('OK')
"
```

---

## 六、开发约定

- **配置集中**：阈值在 `config.py`，禁止魔术数字
- **TDD**：先测试 → 验证失败 → 实现 → Commit
- **pyautogui 安全**：`FAILSAFE=False`，关闭时确保 `mouseUp()`

---

## 七、参考文档

| 文件 | 用途 |
|------|------|
| `docs/design.md` | 设计原理：架构决策、手势规则、阈值来源、UI 布局、异常处理 |
| `README.md` | 用户面向：功能介绍、安装步骤、使用方法 |
| `airmouse/config.py` | 所有配置常量的**唯一真相源**（阈值、灵敏度、EMA 等） |

**找设计决策 → design.md。找怎么用 → README。找参数值 → config.py。**
