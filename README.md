# AirMouse 🖱️✋

**用摄像头手势识别控制 Linux 桌面光标** —— 多方案实验项目。

不同实现方案各自独立目录，互不干扰，方便对比实验。

## 项目结构

```
AirMouse/
├── CLAUDE.md                    # 项目框架说明（给 AI）
├── README.md                    # 本文件（给人看）
├── .gitignore
└── strategy-v1-mediapipe/       # ✅ 独立子项目（自带 CLAUDE.md + README.md + 代码）
```

## 当前方案

| 目录 | 方案 | 状态 |
|------|------|------|
| [strategy-v1-mediapipe/](./strategy-v1-mediapipe/) | MediaPipe Hands 21 关键点 → 食指移动、捏合点击、握拳暂停 | ✅ 完成 |

## 快速开始

```bash
git clone https://github.com/Liujingze11/AirMouse.git
cd AirMouse/strategy-v1-mediapipe

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

> 详细说明见各策略目录内的 README。

## 后续方向

不同手势模型、不同交互姿势（如"握鼠标仅动食指拇指"）、不同映射策略等，各自独立 `strategy-vN-xxx/` 目录。

## 许可证

MIT
