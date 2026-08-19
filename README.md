# MuMu 模拟器 + ADB + OpenCV 页面自动化学习示例

一个面向本地学习 / 测试环境的 Python 页面自动化示例。它演示了如何结合 **MuMu 模拟器（Android）**、**ADB** 与 **OpenCV 模板/颜色识别**，实现一个完整的“页面状态判断 → 触发点击 → 异常崩溃恢复”状态机：在游戏商城中监控某个商品的“公示期”倒计时，公示期结束后自动完成购买与确认操作。

> ⚠️ **重要声明**
>
> 本项目仅用于**你有权测试的本地/测试环境**（自有账号、沙盒页面、测试商品）。请遵守目标应用的服务条款与调用限制；**不提供**任何绕过验证码、访问控制、反自动化机制或其他平台保护措施的能力。请勿用于真实商城下单或任何违反平台规则的行为。

## 功能特性

- **公示期状态检测**：通过 HSV 颜色检测确认目标商品仍被选中（青色高亮框），并检测商品卡片右下角的红色“公示期”文字是否消失，多帧确认后才判定公示期结束。
- **购买触发**：支持普通公示期购买与“叠挂”购买（商品详情弹窗 → 再点下一层确认弹窗），状态机保证每一步只触发一次。
- **崩溃自动恢复**：检测到游戏异常退出回到 MuMu 桌面时，按页面模板与最低等待时间自动重新进入“我的关注”页面，再恢复公示期检测。
- **工程化细节**：ADB 封装、模板匹配、指数退避重试、点击防抖、冷却时间、ROI 裁剪加速、双通道日志、事件截图。

## 目录结构

```
.
├── main.py                  # 主程序：状态机、ADB 封装、模板匹配、重试、防抖、日志
├── config.json              # 全部配置：设备、轮询、阈值、模板路径、ROI、恢复参数
├── requirements.txt         # Python 依赖
├── templates/               # 模板图片（购买/确认按钮、恢复流程页面）
│   ├── purchase_button.png          # 普通购买按钮模板
│   ├── stacked_purchase_button.png  # 叠挂弹窗“购买”按钮模板
│   ├── confirm_order_button.png     # 确认弹窗“确定”按钮模板
│   └── recovery/                    # 崩溃恢复流程页面模板
└── tools/
    └── build_recovery_templates.py  # 从全屏截图中批量裁剪恢复模板
```

## 环境要求

- Windows 10/11（依赖 Win32 窗口捕获，Linux/macOS 仅可使用 ADB 模式）
- Python 3.11+
- [MuMu 模拟器 12](https://www.mumu.com/)（分辨率 1920×1080）
- MuMu 自带 ADB（或任意 Android platform-tools 中的 adb）

## 安装与配置

```powershell
cd <本项目目录>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
adb devices   # 确认设备已连接
```

依赖只有三个：

```
numpy>=1.24,<3
opencv-python>=4.8,<5
mss>=9,<11
```

**ADB 配置**：`config.json` 中 `adb.path` 默认指向 `D:\MuMu\nx_device\15.0\shell\adb.exe`，设备序列号为 `127.0.0.1:7555`。如果你的 MuMu 安装在其他位置，请修改这两个字段。

## 使用方法

### 1. 抓取当前屏幕（校准用）

```powershell
python main.py --capture work\full_screen.png
```

脚本会将当前模拟器画面保存为 PNG，供你核对模板与 ROI 是否对准。

### 2. 先以 dry_run 模式试运行

`config.json` 中 `runtime.dry_run` 默认建议保持 `true`：

```json
"dry_run": true
```

该模式**只识别并在日志中打印“将点击”的坐标，不会向模拟器发送任何点击**，便于核对坐标和模板是否准确。

### 3. 正式运行

只有在你有明确授权的本地测试页面、且确认识别无误后，才可将 `dry_run` 改为 `false`：

```powershell
python main.py --config config.json
```

- 停止运行：`Ctrl+C`
- 日志默认写入 `logs/automation.log`（含时间、匹配分数、状态转换、点击坐标、错误原因）

## 配置说明（config.json）

| 模块 | 说明 |
|---|---|
| `adb` | adb 路径、设备序列号、是否启动时连接、命令超时、截图模式（`raw`/`png`） |
| `capture` | 截图来源（`window` 窗口捕获 / `adb`）、窗口标题关键字、窗口几何刷新周期、最小化时是否自动恢复、窗口失败是否降级到 ADB |
| `screen` | 期望的屏幕尺寸（1920×1080），尺寸不符会中止检测 |
| `runtime` | `dry_run`、`auto_confirm`、各点击坐标、轮询间隔、防抖/冷却时间、错误重试参数、单次运行最大下单数（`max_orders_per_run`，0 表示不限） |
| `recovery` | 崩溃恢复开关与各步骤的最低等待时间、点击坐标、超时 |
| `logging` | 日志级别、日志文件路径、是否保存事件截图 |
| `status_detection` | 青色选中框、红色公示期文字、“我的关注”选中态的 HSV 检测区域与阈值 |
| `templates` | 各按钮模板路径、匹配阈值、搜索 ROI |

### 公示期判定逻辑

倒计时数字不断变化，因此**不使用文字模板**，改用闭锁检测：

1. 在 `selection_guard.region` 检测青色高亮像素 —— 目标商品未被选中时不购买；
2. 在 `publicity_red_text.region` 检测红色像素 —— 达到 `minimum_pixels` 说明公示期文字仍在；
3. `required_consecutive_publicity_absent_matches` 次连续检测到红字消失，判定公示期结束；
4. 匹配购买按钮 → 点击 → 等待 `post_purchase_wait_seconds` → 在确认弹窗 ROI 中搜索“确定”按钮。

### 崩溃恢复流程

`recovery.enabled=true` 时，若目标商品选中框消失，程序每 2 秒检查一次 MuMu 桌面上的游戏图标，检测到后按“游戏图标 → 选服 → 角色 → 开始 → 商业街 → 店铺 → 我的关注”的顺序逐步恢复。每一步都要求**先达到最低等待时间，再由 OpenCV 模板匹配确认页面**，不会只按固定时间盲点。页面模板位于 `templates/recovery/`，等待参数位于 `config.json` 的 `recovery` 部分。

## 重新制作模板与校准 ROI

页面分辨率或样式变化后，需要重新裁剪模板并校准检测区域：

1. 运行 `python main.py --capture work\full_screen.png` 抓取当前屏幕；
2. 用任意图像工具从截图中裁剪按钮，保存为 `templates/*.png`（购买按钮、叠挂购买按钮、确认按钮，恢复流程模板可用 `tools/build_recovery_templates.py` 批量生成）；
3. 在 `config.json` 中重新设置各模板的 `region` 与颜色检测的 ROI（坐标原点为截图左上角，单位像素）；
4. 先在 `dry_run=true` 下验证识别与坐标，确认无误后再考虑关闭。

## 防误触与稳健性设计

- 购买前要求青色选中框存在，避免页面未加载或选错商品时触发；
- 状态机互斥：购买阶段不再处理购买，确认阶段不再处理确认；
- 购买/确认各自独立冷却 + 全局最小点击间隔；
- 确认弹窗超时后回到状态检测，不会无限点击；
- ADB 断开、截图解码失败、ROI 越界等异常按指数退避重试，连续失败后暂停；
- 模板匹配只裁剪按钮 ROI 再转灰度处理，不做全屏匹配，速度更快。

## 相关说明

- 极速配置下事件截图已关闭（`save_event_screenshots=false`）；排查坐标时可临时改回 `true`。
- 使用窗口捕获模式时必须保持 MuMu 窗口可见且不被遮挡；窗口被遮挡或最小化时自动降级为 ADB 截图。
- 模板匹配不具备 OCR 语义理解能力，按钮颜色、字体、缩放或主题变化都会影响分数，页面改版后应重新校准。

## License

本项目仅作学习交流用途，代码可按需修改与使用（遵守上述合规声明）。
