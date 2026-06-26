# AirMouse — 项目上下文

> 多方案实验项目框架。各策略目录是独立子项目，各自带有 `CLAUDE.md` + `README.md` + 代码。
> 最后更新：2026-06-26

---

## 一、项目一句话

**用摄像头手势识别控制 Linux 桌面光标的多方案实验项目。** 不同方案（追踪算法、交互姿势、映射策略）各自独立目录，互不干扰，方便对比。

---

## 二、策略目录约定

```
AirMouse/
├── CLAUDE.md                 # 本文件（框架层面）
├── README.md                 # 项目概述（给人看）
├── .gitignore
└── strategy-v1-mediapipe/    # 独立子项目
    ├── CLAUDE.md             # 该方案的 AI 上下文
    ├── README.md             # 该方案的用户文档
    ├── docs/                 # 该方案的设计文档
    ├── main.py
    └── ...
```

- 每个策略目录是自含的独立项目：自带 `CLAUDE.md`、`README.md`、代码、测试、文档。
- 命名规范：`strategy-v{序号}-{简短描述}`
- 目录之间互不依赖、互不干扰。

---

## 三、当前策略

| 目录 | 方案 | 状态 |
|------|------|------|
| [strategy-v1-mediapipe/](./strategy-v1-mediapipe/) | MediaPipe Hands 21 关键点 → 单手手势控制光标 | ✅ v1 完成 |

---

## 四、开发约定

- **Commit 风格**：`feat:` / `fix:` / `chore:` / `docs:`
- **TDD**：先测试 → 验证失败 → 实现 → Commit
- **配置集中**：各方案的阈值/参数集中在该方案的 `config.py`，禁止魔术数字

---

## 五、会话日志

> 每次会话结束后追加一行。

### 2026-06-26
- 重组项目结构：各策略目录为独立子项目，自带 CLAUDE.md + README.md
- v1 设计文档和 README 移入 strategy-v1-mediapipe/，顶层仅保留框架概述
- 删除顶层 docs/ 目录（内容已归入策略目录）
